SELECT DISTINCT c.id, c.geom, c.name, c.population, c.crime_inde, c.university 
FROM cities c
JOIN counties co ON ST_Within(c.geom, co.geom) 
JOIN recareas r ON ST_DWithin(ST_Transform(c.geom, 2272), ST_Transform(r.geom, 2272), 52800)
JOIN interstates i ON ST_DWithin(ST_Transform(c.geom, 2272), ST_Transform(i.geom, 2272), 105600)
WHERE co.no_farms87 > 500 
  AND co.age_18_64 >= 25000 
  AND c.crime_inde <= 0.02 
  AND co.pop_sqmile < 150 
  AND c.university > 0
