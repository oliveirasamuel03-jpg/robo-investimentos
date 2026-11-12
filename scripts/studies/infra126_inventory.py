"""Read-only AST inventory of the pinned repository; no application imports/DB.

Includes explicit sinks and statically reachable callers. Dynamic dispatch and
SQL supplied outside the repository remain explicit limits, not proof of absence.
Only source locations, symbols and namespace expressions are exported, no payloads.
"""
from __future__ import annotations
import argparse,ast,csv,hashlib,io,json,re,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE='cbd9a17fe40648cd6843792f7842c66da3f915d2'
API_NAMES={'save_json_state','update_json_state','append_json_row','replace_json_rows','append_table_row','replace_table',
 'save_bot_state','update_bot_state','reset_state','reset_paper_state','save_paper_state','append_trade_report',
 'replace_trade_reports','replace_storage_table','append_csv_row','request_stop','acknowledge_stop','start_robot',
 'begin_intent','record_unit_result','post_commit_failure','reconcile'}
FS={'write_text','write_bytes','unlink','rename','to_file'}
SQL_WRITE={'INSERT','UPDATE','DELETE','TRUNCATE','CREATE','ALTER','DROP','GRANT','REVOKE','COPY'}


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def module(path):return path.removesuffix('.py').replace('/','.').removesuffix('.__init__')
def name(node):
 if isinstance(node,ast.Name):return node.id
 if isinstance(node,ast.Attribute):return name(node.value)+'.'+node.attr
 return ast.unparse(node)
def group(path):
 if path.startswith('tests/'):return 'TEST_ONLY'
 if path=='bot_engine.py' or path in {'bot_runner.py','scheduler.py','core/scheduler.py'}:return 'LEGACY_LOCAL'
 if path=='scripts/sql/new_history_schema.sql':return 'EMPTY_DB_INITIALIZATION'
 if path.startswith('scripts/prototypes/'):return 'REJECTED_PROTOTYPE_LOCAL'
 if path=='scripts/local_synthetic_backup.py':return 'SYNTHETIC_BACKUP'
 if path in {'scripts/audit_database_readonly.py','scripts/local_web.py'}:return 'READ_ONLY_OR_LAUNCHER'
 if path.startswith('scripts/'):return 'STUDY_SCRIPT'
 if path=='core/persistence.py':return 'PERSISTENCE'
 if path=='core/commit_protocol.py':return 'COMMIT_PROTOCOL'
 if path=='core/stop_control.py':return 'STOP_PROTOCOL'
 if path=='core/retention.py':return 'RETENTION'
 if path=='core/auth/users_store.py':return 'AUTH'
 if path.startswith('pages/') or path=='app.py':return 'UI'
 if path.startswith('workers/'):return 'WORKER'
 if path in {'paper_trading_engine.py','engines/trader_engine.py','core/trader_reports.py'}:return 'FINANCIAL_UNIT'
 if path=='core/state_store.py':return 'STATE_STORE'
 return 'DELEGATED_OR_DIAGNOSTIC'


class Scan(ast.NodeVisitor):
 def __init__(self,path,tree):
  self.path=path;self.mod=module(path);self.stack=[];self.withs=[];self.imports={};self.stars=[]
  self.nodes={};self.calls=[];self.sinks=[];self.visit(tree)
 def visit_Import(self,node):
  for alias in node.names:self.imports[alias.asname or alias.name.split('.')[0]]=alias.name if alias.asname else alias.name.split('.')[0]
 def visit_ImportFrom(self,node):
  target=node.module or ''
  if node.level:target='.'.join(self.mod.split('.')[:-node.level])+('.'+target if target else '')
  for alias in node.names:
   if alias.name=='*':self.stars.append(target)
   else:self.imports[alias.asname or alias.name]=target+'.'+alias.name
 def visit_FunctionDef(self,node):
  self.stack.append(node.name);key=self.mod+'.'+'.'.join(self.stack)
  self.nodes[key]={'path':self.path,'line':node.lineno,'symbol':key,'decorators':[ast.unparse(d) for d in node.decorator_list],'group':group(self.path),'contract_ref':group(self.path)}
  self.generic_visit(node);self.stack.pop()
 visit_AsyncFunctionDef=visit_FunctionDef
 def visit_ClassDef(self,node):self.stack.append(node.name);self.generic_visit(node);self.stack.pop()
 def visit_With(self,node):
  before=len(self.withs);self.withs.extend(ast.unparse(i.context_expr) for i in node.items)
  self.generic_visit(node);del self.withs[before:]
 visit_AsyncWith=visit_With
 def visit_Call(self,node):
  raw=name(node.func);head,*tail=raw.split('.')
  resolved=self.imports.get(head,self.mod+'.'+head if not tail else head)
  if tail:resolved+='.'+'.'.join(tail)
  caller=self.mod+'.'+'.'.join(self.stack) if self.stack else self.mod+'.<module>'
  if caller not in self.nodes:self.nodes[caller]={'path':self.path,'line':1,'symbol':caller,'decorators':[],'group':group(self.path),'contract_ref':group(self.path)}
  self.calls.append({'caller':caller,'callee':resolved,'raw':raw,'path':self.path,'line':node.lineno})
  leaf=raw.split('.')[-1];kind=None;namespace=None;sql_verbs=[];tables=[]
  if leaf in API_NAMES:
   kind='python_mutation_or_control_api'
   namespace=ast.unparse(node.args[0])[:180] if node.args and leaf in {'save_json_state','update_json_state','append_json_row','replace_json_rows','append_table_row','replace_table','replace_storage_table','append_csv_row'} else None
  elif leaf in FS or raw in {'os.replace','os.remove','shutil.move','shutil.copy','shutil.copy2','shutil.copyfile'}:kind='filesystem_sink'
  elif leaf in {'open','NamedTemporaryFile'}:
   modes=[a.value for a in node.args if isinstance(a,ast.Constant) and isinstance(a.value,str)]
   modes += [k.value.value for k in node.keywords if k.arg=='mode' and isinstance(k.value,ast.Constant)]
   if any(any(c in str(v) for c in 'wax+') and len(str(v))<=4 for v in modes):kind='file_open_write'
  elif leaf in {'execute','executemany'}:
   first=node.args[0] if node.args else None
   if isinstance(first,ast.Constant) and isinstance(first.value,str):text=first.value
   elif isinstance(first,ast.JoinedStr):text=''.join(v.value if isinstance(v,ast.Constant) and isinstance(v.value,str) else '<expr>' for v in first.values)
   else:text=''
   sql_verbs=sorted(set(re.findall(r'\b(?:INSERT|UPDATE|DELETE|TRUNCATE|CREATE|ALTER|DROP|GRANT|REVOKE|COPY)\b',text.upper())))
   tables=sorted(set(re.findall(r'\b(?:INTO|FROM|UPDATE|TABLE)\s+([a-z_][a-z_0-9.]*)',text,re.I)))
   kind=('sql_read_with_row_lock' if 'FOR UPDATE' in text.upper() else 'sql_read_or_session') if text.lstrip().upper().startswith(('SELECT','SHOW')) else 'sql_write_or_ddl' if sql_verbs else 'sql_read_or_session' if text else 'sql_dynamic_review_required'
  elif leaf=='to_csv':
   durable=bool(node.args) or any(k.arg=='path_or_buf' and not (isinstance(k.value,ast.Constant) and k.value.value is None) for k in node.keywords)
   kind='csv_file_sink' if durable else 'in_memory_csv_export_not_writer'
  if kind:
   self.sinks.append({'path':self.path,'line':node.lineno,'caller':caller,'callee':resolved,'kind':kind,
      'group':group(self.path),'namespace_expression':namespace,'with_contexts':list(self.withs),
      'sql_verbs':sql_verbs,'sql_tables':tables,'contract_ref':group(self.path),
      'confidence':'explicit_source_sink' if kind!='sql_dynamic_review_required' else 'manual_review_required'})
  self.generic_visit(node)


CONTRACTS={
 'PERSISTENCE': ['app_state[*]; app_rows[*]; local JSON/CSV','init/CAS/append/replace DELETE+INSERT','PG global→namespace→row; local flock','strict JSON/finite/duplicate-key; CAS and STOP/gate guards','active PG unit/savepoints; local per-file only','no trusted namespace revision','SQL/old writers can bypass application guards'],
 'COMMIT_PROTOCOL': ['paper_commit_control; paper_cycle_receipt:<id>','PREPARED/receipt/verified resolution/diagnostic','global→namespace; local file locks','identity + full financial manifest and contract diagnostics','intent independent; receipt same financial commit; resolution separate','none; hashes are evidence, not counters','older binaries must not mutate protocol namespaces'],
 'STOP_PROTOCOL': ['app_state[bot_state].stop_control/status','request/ACK/explicit START','global→bot_state; local bot path lock','control revision/CAS; no generic STOP clearing','standalone request commit+readback; ACK unit','control revision tuple, not integrity revision','legacy saves/SQL do not honor explicit transition authority'],
 'STATE_STORE': ['app_state[bot_state]; runtime tables','init/normalize/patch/CAS/reset/logs','persistence global→namespace','CAS/strict JSON/STOP/financial guards','individual API or enclosing unit; resets can span multiple calls','none','raw JSON overwrite/old reset unsafe; reads may write'],
 'FINANCIAL_UNIT': ['paper_state; bot_state mirror; paper_trades; trader_orders; trade_reports','draft commit/BUY/SELL/mirror/reports/reset','global financial unit + namespaces; no feed under lock','PAPER/STOP/risk/CAS/identity/full proof','cash+position+trade+report+receipt same PG commit; standalone reset is separate path','none','manual PAPER and trader finalize have different mirror contracts'],
 'RETENTION': ['runtime reports/orders/logs; monthly archives; weekly tables/summaries','normalize/dedupe/copy/replace; catalog','global transaction via decorator/worker','gate guards on financial runtime; no independent integrity certification','PG archive+runtime; worker wraps catalog/log; local not multi-file atomic','none','receipt seq/payload/ns change; no-op also rewrites; archive catalog matters'],
 'AUTH': ['app_state[auth_users_store]; local auth_users.json','create/password/role/disable/bootstrap','global→auth namespace; bcrypt outside lock','credential structure/roles/duplicates/admin continuity','update_json_state PG unit or local file lock','none','application admin is not a PostgreSQL administrator'],
 'UI': ['delegated bot state/control/financial APIs/users','read-init/config/manual cycle/reset/reconcile','inherited; no separate UI database lock','role controls + shared mutation helpers/CAS; never blind retry','depends on called API; session intent is not durable STOP','none','stale full saves conflict; UI is not an additional financial authority'],
 'WORKER': ['delegated financial/control/state/retention/logs','one cycle/heartbeat/maintenance/post-commit diagnostics','inherited global units; retention outer unit','gate/STOP/PAPER; outcome-specific handling','financial unit; separate diagnostics; maintenance+catalog atomic PG','none','do not mix worker versions or start extra worker'],
 'LEGACY_LOCAL': ['legacy_runtime_config.json; legacy_bot_state.json; legacy_bot_logs.jsonl','direct write/reset/append','no current global/CAS/flock protocol','json.loads fallback to defaults; separate legacy mode','no multi-file/PG transaction','none','incompatible independent ledger; separate file paths, not same canonical file'],
 'SYNTHETIC_BACKUP': ['own app_state/app_rows + dump + newly created restore DB','STOP/export/pg_dump/restore/quarantine','source session advisory lock + RR export snapshot','owned label/port/user/name; exact inventory; STOP/gate preserved','consistent source dump; restore single transaction; quarantine after comparison','no portability of cached proof','not recovery of old DB; no automatic RUNNING or gate release'],
 'EMPTY_DB_INITIALIZATION': ['public.app_state; public.app_rows; indexes/sequence','CREATE only on dedicated empty database','schema/script transaction and refusal of existing relations','empty-user-relations check; no seed/import/worker','BEGIN/COMMIT DDL','none','not an upgrade/migration of existing historical database'],
 'REJECTED_PROTOTYPE_LOCAL': ['same protocol metadata only; local revision study table','read-only plan/exact fence/publish metadata; local DDL','RR + global + broad SQL fence (rejected)','raw equality/identity; counters not authoritative','local synthetic only; never runtime integration','toy transactional revisions can be bypassed','rejected for STOP/contention/retention; no financial retry'],
 'STUDY_SCRIPT': ['synthetic DB/files/evidence or launched helper','local checks/reproductions/exports','script-specific; manual review','local isolation/explicit URL contracts vary','not an approved application writer','none','never run old-code reproduction against real data'],
 'READ_ONLY_OR_LAUNCHER': ['audit SELECT or launched application APIs','read-only inventory/launcher','audit RR READ ONLY; launch delegates','audit parser is not strict financial proof','read-only SQL or delegated startup','none','do not confuse SELECT tool with initializing application loader'],
 'TEST_ONLY': ['owned disposable DB/temp files only','fault injection/SQL/DDL/backup/cleanup','test-specific barriers/locks/fixtures','explicit local opt-in/PAPER; synthetic only','case-specific incl deliberate rollback/corruption','study only','not runtime writer permissions'],
 'DELEGATED_OR_DIAGNOSTIC': ['delegated bot_state/logs/cache or read-init','status/study/alerts/diagnostics/initialization','inherited or explicit filesystem path','depends on destination API; diagnosis has no entry authority','individual API or enclosing caller transaction','none','call graph overapproximation; inspect namespace/role before approval'],
}


def build():
 paths=git('ls-tree','-r','--name-only',BASE).decode().splitlines()
 scans={};sources=[];sql=[]
 for path in paths:
  if not path.endswith(('.py','.sql','.sh')):continue
  content=git('show',f'{BASE}:{path}');sources.append({'path':path,'sha256':hashlib.sha256(content).hexdigest()})
  if path.endswith('.py'):scans[module(path)]=Scan(path,ast.parse(content,filename=path))
  elif path.endswith('.sql'):
   text=re.sub(r'--[^\n]*','',content.decode())
   sql.append({'path':path,'group':group(path),'verbs':sorted(set(re.findall(r'\b(?:INSERT|UPDATE|DELETE|TRUNCATE|CREATE|ALTER|DROP|GRANT|REVOKE|COPY)\b',text.upper()))),'manual_review_required':True})
 nodes={k:v for scan in scans.values() for k,v in scan.nodes.items()}
 def resolve(target,seen=None):
  seen=seen or set()
  if target in seen:return target
  seen.add(target)
  if target in nodes:return target
  mod,_,symbol=target.rpartition('.')
  scan=scans.get(mod)
  if not scan:return target
  if symbol in scan.imports:return resolve(scan.imports[symbol],seen)
  for star in scan.stars:
   candidate=resolve(star+'.'+symbol,seen.copy())
   if candidate in nodes:return candidate
  return target
 edges=[{**edge,'resolved_callee':resolve(edge['callee'])} for scan in scans.values() for edge in scan.calls]
 sinks=[sink for scan in scans.values() for sink in scan.sinks]
 writers={s['caller'] for s in sinks if s['kind'] not in {'sql_read_or_session','sql_read_with_row_lock','in_memory_csv_export_not_writer'}}
 direct=set(writers)
 changed=True
 while changed:
  additions={e['caller'] for e in edges if e['resolved_callee'] in writers}-writers
  changed=bool(additions);writers.update(additions)
 return {'scope':'STUDY_ONLY_SOURCE_AUDIT','repository':'oliveirasamuel03-jpg/robo-investimentos','base_commit':BASE,
  'contracts_A_to_G':{k:dict(zip(['A_namespace','B_change','C_lock','D_validation','E_transaction','F_revision','G_incompatible_writer_risk'],v)) for k,v in CONTRACTS.items()},
  'source_files':sources,'sql_files':sql,'explicit_sinks':sorted(sinks,key=lambda x:(x['path'],x['line'])),
  'potential_writers':[dict(nodes[k],direct_sink=k in direct) for k in sorted(writers)],
  'call_edges_to_potential_writers':[e for e in edges if e['resolved_callee'] in writers],
  'limits':['Static overapproximation, not proof of execution or universal completeness.',
            'Dynamic SQL/callbacks/monkeypatch/external administrative SQL and dynamic file modes need manual review.',
            'loaders can initialize/write; a name containing read/load is not a read-only guarantee.',
            'No module imported, no DB accessed, no runtime activated.']}


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'docs/retomada/evidencias/infra126-writers.json')
 args=parser.parse_args();result=build();args.output.parent.mkdir(parents=True,exist_ok=True)
 args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 out=io.StringIO();writer=csv.DictWriter(out,lineterminator='\n',fieldnames=['path','line','symbol','group','direct_sink','decorators','A_namespace','B_change','C_lock','D_validation','E_transaction','F_revision','G_incompatible_writer_risk']);writer.writeheader()
 for row in result['potential_writers']:writer.writerow({**{k:row[k] for k in ('path','line','symbol','group','direct_sink','decorators')},**result['contracts_A_to_G'][row['group']]})
 args.output.with_suffix('.csv').write_text(out.getvalue())
 print(json.dumps({'source_files':len(result['source_files']),'explicit_sinks':len(result['explicit_sinks']),
                   'potential_writer_functions':len(result['potential_writers']),'output':str(args.output)}))

if __name__=='__main__':main()
