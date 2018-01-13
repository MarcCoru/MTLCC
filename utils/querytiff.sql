COPY(
SELECT encode(ST_astiff(ST_UNION(ST_CLIP(r.rast, t.geom)),ARRAY[1,2,3]),'hex')
        from
            bavaria r, tiles240 t
        where
            t.id=1 and
            ST_INTERSECTS(r.rast,t.geom) and
            r.type='10m' and
            r.level='L1C' and
            date='2016-08-28'::date
)
TO STDOUT;
