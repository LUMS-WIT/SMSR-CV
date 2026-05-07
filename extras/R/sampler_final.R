library(terra)
library(stringr)
library(fs)


# polys <- vect("D:/SM-DeepLearning/datasets/Central valley/shapefile/CV_samples/sampler_1km_6X6.shp")
# plot(polys)
# 
# n <- nrow(polys)
# 
# top_left_coords <- data.frame(
#   id  = 0:(n - 1),   # keep your 0-based IDs
#   lon = numeric(n),
#   lat = numeric(n)
# )
# 
# for (i in seq_len(n)) {
#   e <- ext(polys[i])
#   top_left_coords$lon[i] <- e$xmin
#   top_left_coords$lat[i] <- e$ymax
# }
# 
# top_left_coords

# 8 X 8
# id       lon      lat
# 1  0 -121.9621 38.71544
# 2  1 -121.0105 37.66866
# 3  2 -119.6782 35.95576
# 4  3 -120.2492 36.81221
# 5  4 -122.3427 39.66706

# # Define reference top-left coordinates
# top_left_coords <- data.frame(
#   id = 0:4,
#   lon = c(-122.3427, -121.9621, -121.010, -120.2492, -119.6782),
#   lat = c(39.66706, 38.71544, 37.66866, 36.81221, 35.95576)
# )
# 
# 7 X 7
# id       lon      lat
# 1  0 -121.7718 38.52512
# 2  1 -121.0105 37.57350
# 3  2 -119.6782 35.86060
# 4  3 -120.1540 36.71705
# 5  4 -122.2476 39.47673


# top_left_coords <- data.frame(
#   id = 0:4,
#   lon = c(-122.2476, -121.7718, -121.0105, -120.1540, -119.6782),
#   lat = c(39.47673, 38.52512, 37.57350, 36.71705, 35.86060)
# )

# 6 x 6
# id       lon      lat
# 1  0 -121.7718 38.42996
# 2  1 -120.9153 37.47834
# 3  2 -119.5831 35.76543
# 4  3 -120.0589 36.62189
# 5  4 -122.1524 39.38204


top_left_coords <- data.frame(
  id = 0:4,
  lon = c(-122.1524, -121.7718, -120.9153, -120.0589, -119.5831),
  lat = c(39.38204, 38.42996, 37.47834, 36.62189, 35.76543)
)

root_dir <- "D:/SM-DeepLearning/datasets/Central valley"


# Define input directories
in_root_1km <- file.path(root_dir,"SMAP-P-E-1km-AM/SMAP-P-E-1km-AM")
# in_root_3km <- file.path(root_dir,"SMAP-P-E-1km-AM/SMAP-P-E-3km-AM")
in_root_9km_r <- file.path(root_dir,"SMAP-P-E-1km-AM/SMAP-P-E-9km-AM")
in_root_9km <- file.path(root_dir,"SMAP-P-E-9km-AM")
in_root_1km_r <- file.path(root_dir,"SMAP-P-E-1km-AM-r")

in_root_1km_lst <- file.path(root_dir, "datasets/1km/LST")

# Output directory
# out_root <- "D:/SM-DeepLearning/datasets/Central valley/training/SMAP-Resampled"
out_root <- "D:/SM-DeepLearning/datasets/Central valley/training"
dir_create(file.path(out_root, "1km"))
# dir_create(file.path(out_root, "3km"))
dir_create(file.path(out_root, "9km"))
dir_create(file.path(out_root, "9km-r"))
dir_create(file.path(out_root, "1km-lst"))
dir_create(file.path(out_root, "1km-r"))

# List all 1km files and assume corresponding 3km and 9km files exist
tif_files_1km <- list.files(in_root_1km, pattern = "\\.tif$", recursive = TRUE, full.names = TRUE)

# tif_files_1km <- tif_files_1km[1:3]
# tif_files_1km

# Utility: get corresponding path in other resolutions
get_corresponding_file <- function(file_1km, from_root, to_root, res_tag_from, res_tag_to) {
  rel_path <- path_rel(file_1km, start = from_root)
  rel_path_new <- str_replace(rel_path, res_tag_from, res_tag_to)
  file.path(to_root, rel_path_new)
}


extract_block_by_indices <- function(r, lon, lat, width_px = 8, height_px = 8) {
  res_x <- res(r)[1]
  res_y <- res(r)[2]
  
  # Move corner coords to pixel center
  lon_adj <- lon + res_x / 2
  lat_adj <- lat - res_y / 2
  
  cell <- cellFromXY(r, cbind(lon_adj, lat_adj))
  rc <- rowColFromCell(r, cell)
  
  start_row <- rc[1]
  start_col <- rc[2]
  end_row <- start_row + height_px - 1
  end_col <- start_col + width_px - 1
  
  # Convert row/col back to XY coords
  top_left     <- xyFromCell(r, cellFromRowCol(r, start_row, start_col))
  bottom_right <- xyFromCell(r, cellFromRowCol(r, end_row, end_col))
  
  # Build extent using half-cell shifts
  xmin <- top_left[1]   - res_x / 2
  xmax <- bottom_right[1] + res_x / 2
  ymax <- top_left[2]   + res_y / 2
  ymin <- bottom_right[2] - res_y / 2
  
  crop(r, ext(xmin, xmax, ymin, ymax))
}

# Main loop
for (file_1km in tif_files_1km) {
  # file_3km <- get_corresponding_file(file_1km, in_root_1km, in_root_3km, "1km", "3km")
  file_9km_r <- get_corresponding_file(file_1km, in_root_1km, in_root_9km_r, "1km", "9km")
  file_1km_r <- get_corresponding_file(file_1km, in_root_1km, in_root_1km_r, "1km", "1km")
  file_9km <- get_corresponding_file(file_1km, in_root_1km, in_root_9km, "1km", "9km")
  file_1km_lst<- get_corresponding_file(file_1km, in_root_1km, in_root_1km_lst, "SMAP-E", "MYD11A1-LST")
  
  if (!file.exists(file_9km) || !file.exists(file_1km_lst)) {
    message("Skipping due to missing corresponding files: ", file_1km)
    next
  }

  # if (!file.exists(file_3km) || !file.exists(file_9km)) {
  #   message("Skipping due to missing corresponding files: ", file_1km)
  #   next
  # }
  
  r_1km <- rast(file_1km)
  # r_3km <- rast(file_3km)
  r_9km <- rast(file_9km)
  r_9km_r <- rast(file_9km_r)
  r_1km_r <- rast(file_1km_r)
  r_1km_lst <- rast(file_1km_lst)
  
  date_tag <- str_extract(basename(file_1km), "\\d{8}")  # e.g., 20150401
  
  for (i in 1:nrow(top_left_coords)) {
    lon <- top_left_coords$lon[i]
    lat <- top_left_coords$lat[i]
    id <- top_left_coords$id[i]
    
    # block_1km <- extract_block_by_indices(r_1km, lon, lat, 72, 72)
    # block_3km <- extract_block_by_indices(r_3km, lon, lat, 24, 24)
    # block_9km <- extract_block_by_indices(r_9km, lon, lat, 8, 8)

    block_1km <- extract_block_by_indices(r_1km, lon, lat, 54, 54)
    # block_3km <- extract_block_by_indices(r_3km, lon, lat, 18, 18)
    block_9km <- extract_block_by_indices(r_9km, lon, lat, 6, 6)
    block_9km_r <- extract_block_by_indices(r_9km_r, lon, lat, 6, 6)
    block_1km_lst <- extract_block_by_indices(r_1km_lst, lon, lat, 54, 54)
    block_1km_r <- extract_block_by_indices(r_1km_r, lon, lat, 54, 54)
        
    # # Skip if any NA present
    # if (
    #   # any(is.na(values(block_1km))) ||
    #   # any(is.na(values(block_3km))) ||
    #   any(is.na(values(block_9km)))
    # ) {
    #   message("Skipped: ", date_tag, " id=", id, " — contains NA")
    #   next
    # }
    na_count <- sum(is.na(values(block_9km)))
    na_count_r <- sum(is.na(values(block_9km_r)))
    na_count_lst <- sum(is.na(values(block_1km_lst)))
    
    if ( (na_count > 1) || (na_count_r > 2) || (na_count_lst > 500)) {
      message("Skipped: ", date_tag, " id=", id,
              " — too many NA (", na_count, ")")
      next
    }
    
        
    # Construct output filenames (overwrite original folder with modified filename)
    # Save to CV/training/{res}/filename_id.tif
    base_name_1km <- str_replace(basename(file_1km), "\\.tif$", paste0("_", id, ".tif"))
    base_name_1km_r <- str_replace(basename(file_1km_r), "\\.tif$", paste0("_", id, ".tif"))
    # base_name_3km <- str_replace(basename(file_3km), "\\.tif$", paste0("_", id, ".tif"))
    base_name_9km <- str_replace(basename(file_9km), "\\.tif$", paste0("_", id, ".tif"))
    base_name_9km_r <- str_replace(basename(file_9km_r), "\\.tif$", paste0("_", id, ".tif"))
    base_name_1km_lst <- str_replace(basename(file_1km_lst), "\\.tif$", paste0("_", id, ".tif"))
    
    writeRaster(block_1km, file.path(out_root, "1km", base_name_1km), overwrite = TRUE)
    writeRaster(block_1km_r, file.path(out_root, "1km-r", base_name_1km_r), overwrite = TRUE)
    # writeRaster(block_3km, file.path(out_root, "3km", base_name_3km), overwrite = TRUE)
    writeRaster(block_9km, file.path(out_root, "9km", base_name_9km), overwrite = TRUE)
    writeRaster(block_9km_r, file.path(out_root, "9km-r", base_name_9km_r), overwrite = TRUE)
    writeRaster(block_1km_lst, file.path(out_root, "1km-lst", base_name_1km_lst), overwrite = TRUE)
    
    message("Saved: ", date_tag, " id=", id)
  }
}  
