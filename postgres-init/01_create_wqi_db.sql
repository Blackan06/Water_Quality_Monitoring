-- Create application database for Water Quality Monitoring
DO
$$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_database WHERE datname = 'wqi_db'
   ) THEN
      PERFORM dblink_exec('dbname=' || current_database(), '');
      EXECUTE 'CREATE DATABASE wqi_db OWNER postgres ENCODING ''UTF8''';
   END IF;
EXCEPTION WHEN undefined_object THEN
   -- If dblink is not available or other issue, just try simple create
   BEGIN
      EXECUTE 'CREATE DATABASE wqi_db OWNER postgres ENCODING ''UTF8''';
   EXCEPTION WHEN duplicate_database THEN
      -- already exists
      NULL;
   END;
END
$$;


