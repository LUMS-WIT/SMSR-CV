library(terra)
library(stringr)
library(fs)

scaling <- '1km'
LST<- FALSE

# === Paths ===
root_dir <- "D:/SM-DeepLearning/datasets/Central valley/datasets"

in_root  <- file.path(root_dir, "soil")
if (scaling == '1km'){
  out_root <- file.path(root_dir, "1km/soil")
} else {
  out_root <- file.path(root_dir, "9km/soil")
}
# 
# in_root  <- file.path(root_dir, "DEM")
# if (scaling == '1km'){
#   out_root <- file.path(root_dir, "1km/DEM")
# } else {
#   out_root <- file.path(root_dir, "9km/DEM")
# }
# 
# in_root  <- file.path(root_dir, "NLCD")
# if (scaling == '1km'){
#   out_root <- file.path(root_dir, "1km/NLCD")
# } else {
#   out_root <- file.path(root_dir, "9km/NLCD")
# }

# in_root  <- file.path(root_dir, "MYD11A1_2015_25")
# if (scaling == '1km'){
#   out_root <- file.path(root_dir, "1km/LST")
# } else {
#   out_root <- file.path(root_dir, "9km/LST")
# }

# Reference 9 km raster (grid to align to)

if (scaling == '1km'){
  ref_9km <- rast("D:/SM-DeepLearning/datasets/Central valley/SMAP-P-E-1km-AM/SMAP-E_1km_AM_20150403.tif")
  res(ref_9km)

} else {
  ref_9km <- rast("D:/SM-DeepLearning/datasets/Central valley/SMAP-P-E-9km-AM/SMAP-E_9km_AM_20150403.tif")
  res(ref_9km)
    
  }

# should be c(9000, 9000)

# coarse_res <- "9km"
# fact <- round(res(ref_9km)[1] / 250)  # should be 36 if 250m → 9km


tif_files <- list.files(in_root, pattern = "\\.tif$", recursive = TRUE, full.names = TRUE)
print(length(tif_files))
# tif_files

for (file in tif_files) {
  message("Processing: ", file)

  
  r_fine <- rast(file)
  
  # --- sanity checks ---
  stopifnot(crs(r_fine) == crs(ref_9km))
  
  # --- CONDITIONAL: treat 0 as NA only for LST ---
  if (LST) {
    r_fine[r_fine == 0] <- NA
    # Extract YYYY-MM-DD
    date_str <- str_extract(basename(file), "\\d{4}-\\d{2}-\\d{2}")
    
    # Convert to YYYYMMDD
    date_tag <- gsub("-", "", date_str)
    
    base_name <- paste0(
      "MYD11A1-LST_", scaling, "_AM_", date_tag,".tif"
    )
  } else {
    base_name <- path_rel(file, start = in_root)
  }
  
  # Compute aggregation factor dynamically
  fact <- round(res(ref_9km) / res(r_fine))
  print(fact)

  # 1️⃣ Crop to reference extent
  r_crop <- crop(r_fine, ref_9km)
  
  # 2️⃣ Aggregate using MAJORITY (modal)
  r_agg <- aggregate(
    r_crop,
    fact = fact,
    fun = "modal", #"modal for categorical, mean otherwise
    na.rm = TRUE
  )
  
  # 3️⃣ Snap perfectly to reference grid
  r_final <- resample(
    r_agg,
    ref_9km,
    method = "near"
  )


    # Output path
  # rel_path <- path_rel(file, start = in_root)
#   rel_out_file <- str_replace(rel_path, "1km", coarse_res)
  out_file <- file.path(out_root, base_name)

  dir_create(dirname(out_file))
  writeRaster(r_final, out_file, overwrite = TRUE, wopt = list(gdal = c("COMPRESS=LZW", "TILED=YES")))

  message("Saved: ", out_file)
}
