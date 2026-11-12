-- PROTOTYPE / SYNTHETIC MARKERS ONLY. Not a financial schema or a migration.
-- Executed by the test's NOLOGIN owner in its own disposable database.
CREATE SCHEMA infra126_contract;
CREATE TABLE infra126_contract.authority (
    singleton boolean PRIMARY KEY CHECK (singleton),
    accepted_version integer NOT NULL,
    revision bigint NOT NULL,
    pending boolean NOT NULL
);
INSERT INTO infra126_contract.authority VALUES (true, 1, 0, false);
CREATE TABLE infra126_contract.markers (
    marker_id text PRIMARY KEY,
    content text NOT NULL
);
CREATE FUNCTION infra126_contract.write_marker(
    writer_version integer, expected_revision bigint, marker text, value text
) RETURNS bigint
LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, infra126_contract
AS $$
DECLARE current_contract infra126_contract.authority%ROWTYPE;
BEGIN
    SELECT * INTO current_contract FROM infra126_contract.authority
        WHERE singleton FOR UPDATE;
    IF NOT FOUND OR current_contract.accepted_version <> writer_version THEN
        RAISE EXCEPTION 'INCOMPATIBLE_WRITER';
    END IF;
    IF current_contract.pending THEN
        RAISE EXCEPTION 'PENDING_INTENT';
    END IF;
    IF current_contract.revision <> expected_revision THEN
        RAISE EXCEPTION 'STALE_REVISION';
    END IF;
    INSERT INTO infra126_contract.markers VALUES (marker, value);
    UPDATE infra126_contract.authority SET revision = revision + 1
        WHERE singleton RETURNING revision INTO current_contract.revision;
    RETURN current_contract.revision;
END;
$$;
REVOKE ALL ON FUNCTION infra126_contract.write_marker(integer,bigint,text,text) FROM PUBLIC;
REVOKE ALL ON ALL TABLES IN SCHEMA infra126_contract FROM PUBLIC;
REVOKE ALL ON SCHEMA infra126_contract FROM PUBLIC;
