SELECT 
  LEFT(program_longsynopsis, 512) as synopsis, -- Synopses are sometimes very long
  tags
FROM `ml-sandbox-101.metadata_sky.merlin_movie_series_data`
LIMIT 1000