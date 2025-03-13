# MySql Database DDL

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

-- Increase the radius of the geofence to approximately 30 meters
-- Central point: latitude = 21.1732114, longitude = 79.0961284

-- Calculate the offset in degrees for 30 meters
-- Latitude offset: 30 meters ≈ 0.0002695 degrees
-- Longitude offset: 30 meters ≈ 0.000289 degrees (at latitude 21.1732114)

-- Update the geofence boundary with the new polygon vertices
UPDATE geofence
SET boundary = ST_GeomFromText(
    'POLYGON((
        79.0961284 21.1734809,  -- North: latitude increased by 0.0002695
        79.0964174 21.1732114,  -- East: longitude increased by 0.000289
        79.0961284 21.1729419,  -- South: latitude decreased by 0.0002695
        79.0958394 21.1732114,  -- West: longitude decreased by 0.000289
        79.0964174 21.1734809,  -- Northeast: latitude and longitude increased
        79.0961284 21.1734809   -- Close the polygon
    ))',
    4326
)
WHERE name = 'Geofence Area';  -- Update the specific geofence by name

```
