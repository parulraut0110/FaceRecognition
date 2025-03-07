#MySql Database DDL

```sql

create Database Geofence;

USE Geofence;

CREATE TABLE geofence (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    boundary POLYGON NOT NULL SRID 4326
);

INSERT INTO geofence (name, boundary)
VALUES (
    'Geofence Area',
    ST_GeomFromText(
        'POLYGON((79.0961284 21.1733014, 79.0962184 21.1732114, 79.0961284 21.1731214, 79.0960384 21.1732114, 79.0961734 21.1732564, 79.0961284 21.1733014))',
        4326
    )
);

```
