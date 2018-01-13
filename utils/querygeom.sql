COPY(
SELECT 
	concat('<path d= "',ST_ASSVG(ST_intersection(f.geom,t.geom)),'" />') 
FROM 
	tiles240 t, fields2016 f 
WHERE 
	st_intersects(f.geom,t.geom) and t.id=1
) to STDOUT
