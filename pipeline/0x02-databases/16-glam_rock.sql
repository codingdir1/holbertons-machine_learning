SELECT band_name, IFNULL(split, 2020) - formed AS lifespan
FROM metal_bands
WHERE style LIKE "%Black Metal%"
ORDER BY lifespan DESC
