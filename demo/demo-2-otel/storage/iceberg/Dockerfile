FROM apache/iceberg-rest-fixture:latest

USER root

# Create directory and download PostgreSQL JDBC driver
RUN mkdir -p /usr/lib/iceberg-rest/lib && \
    curl -L -o /usr/lib/iceberg-rest/postgresql.jar https://jdbc.postgresql.org/download/postgresql-42.7.3.jar && \
    chown -R iceberg:iceberg /usr/lib/iceberg-rest

# Optional: log what's inside
RUN ls -l /usr/lib/iceberg-rest/lib

USER iceberg

EXPOSE 8181 
# Run Iceberg RESTCatalog with driver in classpath
CMD ["java", "-cp", "iceberg-rest-adapter.jar:/usr/lib/iceberg-rest/postgresql.jar", "org.apache.iceberg.rest.RESTCatalogServer"]
