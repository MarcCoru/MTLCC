COPY(
SELECT encode(
	st_astiff(
	    st_clip(
	    st_union(
		st_asraster(st_intersection(t.geom,l.geom), 
		    10::float,10::float,st_xmin(t.geom)::float, st_xmax(t.geom)::float,'8BUI',l.labelid,-9999)
	    ),
	    t.geom, -9999)
	   )
,'hex')
    from tiles240 t, fields2016 l 
    where t.id=1 and ST_Intersects(t.geom, l.geom)
    group by t.geom
)
TO STDOUT;
