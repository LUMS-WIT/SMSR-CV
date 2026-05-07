library(terra)
library(stringr)
library(fs)

# NO NEED FOR THIS for LST
# DONE in sampler_final

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

root_dir <- "D:/SM-DeepLearning/datasets/Central valley/datasets/1km"
dataset<- "NLCD"
LST<- FALSE

# Define input directories
in_root <- file.path(root_dir,dataset)

# Output directory

out_root <- "D:/SM-DeepLearning/datasets/Central valley/datasets/1km_samples"
dir_create(file.path(out_root, dataset))
out_path <- file.path(out_root, dataset)

# List all 1km files and assume corresponding 3km and 9km files exist
dataset_files <- list.files(in_root, pattern = "\\.tif$", recursive = TRUE, full.names = TRUE)
len<- length(dataset_files)
print(len)

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

# dataset_files<- dataset_files[20:100]
# print(length(dataset_files))
# Main loop
for (dataset_file in dataset_files) {

  r_9km <- rast(dataset_file)
  
  for (i in 1:nrow(top_left_coords)) {
    lon <- top_left_coords$lon[i]
    lat <- top_left_coords$lat[i]
    id <- top_left_coords$id[i]
    
    block_9km <- extract_block_by_indices(r_9km, lon, lat, 54, 54)

    na_count <- sum(is.na(values(block_9km)))
    
    if (LST) {
      # Extract YYYY-MM-DD
      date_str <- str_extract(basename(dataset_file), "\\d{4}-\\d{2}-\\d{2}")
      
      # Convert to YYYYMMDD
      date_tag <- gsub("-", "", date_str)
      
      item <- date_tag
    } else {
      item <- dataset_file
    }
    
    
    if  (na_count > 500)  {
      message("Skipped: ", item, " id=", id,
              " — too many NA (", na_count, ")")
      next
    }
            

    # Construct output filenames (overwrite original folder with modified filename)
    if (LST) {
      base_name_9km <- paste0(
        "MYD11A1_LST_", date_tag, "_", id, ".tif"
      )
    } else {
      base_name_9km <- str_replace(
        basename(dataset_file),
        "\\.tif$",
        paste0("_", id, ".tif")
      )
    }
    
    
    # coltab(block_9km) <- NULL
    writeRaster(block_9km, file.path(out_path, base_name_9km), overwrite = TRUE)
    
    message("Saved: ", dataset_file, " id=", id)
  }
}  
