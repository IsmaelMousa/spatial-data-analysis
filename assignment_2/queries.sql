SELECT l.id, l.type, l.area, l.owner, l.geom, ST_Centroid(l.geom)

AS centroid,
MIN(ST_Distance(ST_Centroid(l.geom), r.geom)) 
AS min_road_dist

FROM "Landuse" l

JOIN "Roads" r ON ST_DWithin(l.geom, r.geom, 25)

WHERE LOWER(l.type) IN ('un-used', 'agricultural areas', 'commercial lands') 
AND COALESCE(l.area, ST_Area(l.geom)) >= 5000
AND NOT EXISTS (SELECT 1 FROM "Buildings" b
WHERE ST_Intersects(b.geom, l.geom))

GROUP BY l.id, l.type, l.area, l.owner, l.geom;