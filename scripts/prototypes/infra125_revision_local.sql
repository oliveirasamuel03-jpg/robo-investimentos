-- LOCAL SYNTHETIC STUDY ONLY. Never loaded by application schema initialization.
-- Revisions are NOT proof of integrity: privileged writers can bypass triggers.
-- Fails on a second installation; no DROP, reset, or historical data rewrite.
DO $$
BEGIN
 IF current_database() !~ '^atlas_infra12[3]?_' OR current_user <> 'atlas_fixture' THEN
  RAISE EXCEPTION 'Local synthetic fixture database required; prototype DDL refused';
 END IF;
END;
$$;
CREATE TABLE infra125_namespace_revision(namespace TEXT PRIMARY KEY, revision BIGINT NOT NULL);
CREATE FUNCTION infra125_bump(name TEXT) RETURNS void LANGUAGE SQL AS $$
 INSERT INTO infra125_namespace_revision(namespace,revision) VALUES(name,1)
 ON CONFLICT(namespace) DO UPDATE SET revision=infra125_namespace_revision.revision+1;
$$;
CREATE FUNCTION infra125_track() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF TG_OP='TRUNCATE' THEN
  PERFORM infra125_bump('*:truncate'); RETURN NULL;
 END IF;
 IF TG_OP<>'INSERT' THEN PERFORM infra125_bump(OLD.namespace); END IF;
 IF TG_OP='INSERT' THEN PERFORM infra125_bump(NEW.namespace);
 ELSIF TG_OP='UPDATE' AND NEW.namespace IS DISTINCT FROM OLD.namespace THEN
  PERFORM infra125_bump(NEW.namespace);
 END IF;
 RETURN NULL;
END;
$$;
CREATE TRIGGER infra125_state_revision AFTER INSERT OR UPDATE OR DELETE ON app_state FOR EACH ROW EXECUTE FUNCTION infra125_track();
CREATE TRIGGER infra125_rows_revision AFTER INSERT OR UPDATE OR DELETE ON app_rows FOR EACH ROW EXECUTE FUNCTION infra125_track();
CREATE TRIGGER infra125_state_truncate AFTER TRUNCATE ON app_state FOR EACH STATEMENT EXECUTE FUNCTION infra125_track();
CREATE TRIGGER infra125_rows_truncate AFTER TRUNCATE ON app_rows FOR EACH STATEMENT EXECUTE FUNCTION infra125_track();
