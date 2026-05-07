library(terra)
library(fs)
library(stringr)


root_bands <- "D:/SM-DeepLearning/datasets/Central valley/datasets/1km_samples"

dem_dir  <- file.path(root_bands, "DEM")
lulc_dir <- file.path(root_bands, "NLCD")
soil_dir <- file.path(root_bands, "soil")

root_SMAP <- "D:/SM-DeepLearning/datasets/Central valley//training"

lst_dir <- file.path(root_SMAP, "1km-lst")
smap_dir <- file.path(root_SMAP, "1km-r")

out_dir  <- file.path(root_SMAP, "1km-r-m")
dir_create(out_dir)

# Static layers (by ROI id)
get_static_stack <- function(id) {
  
  files <- c(
    file.path(dem_dir,  paste0("CV_SRTM90_Elevation_", id, ".tif")),
    file.path(lulc_dir, paste0("Annual_NLCD_LndCov_2024_", id, ".tif")),
    file.path(soil_dir, paste0("STC_TOPSOIL_", id, ".tif"))
    # file.path(soil_dir, paste0("WCpF2_TOPSOIL_", id, ".tif")),
    # file.path(soil_dir, paste0("WCpF42_TOPSOIL_", id, ".tif")),
    # file.path(soil_dir, paste0("WCres_TOPSOIL_", id, ".tif")),
    # file.path(soil_dir, paste0("WCsat_TOPSOIL_", id, ".tif"))
  )

  r <- rast(files)

  names(r) <- c(
    "DEM",
    "LULC",
    "STC"
    # "WCpF2",
    # "WCpF42",
    # "WCres",
    # "WCsat"
  )

  r
}


get_lst_layer <- function(date, id) {
  lst_file <- file.path(
    lst_dir,
    paste0("MYD11A1-LST_1km_AM_", date, "_", id, ".tif")
  )
  
  if (!file.exists(lst_file)) {
    stop(paste("Missing LST file:", lst_file))
  }
  
  r_lst <- rast(lst_file)
  names(r_lst) <- "LST_AM"
  
  return(r_lst)
}


# Loop over SMAP files and build stacks
smap_files <- list.files(smap_dir, pattern = "\\.tif$", full.names = TRUE)

# smap_files <- smap_files[1:10]

total_files<- length(smap_files)
print(total_files)

i <- 1

for (smap_file in smap_files) {

  message("Processing: (", i,"/",total_files,") ", basename(smap_file))

  # Extract ROI id from filename
  id <- str_extract(basename(smap_file), "_[0-9]+(?=\\.tif)") |> str_remove("_")

  # Load SMAP
  r_smap <- rast(smap_file)
  names(r_smap) <- "SMAP_AM"

  
  # Extract date
  date <- str_extract(basename(smap_file), "\\d{8}")
  
  # Load LST (dynamic, 1 km)
  r_lst <- get_lst_layer(date, id)
                         
  # Load static predictors
  r_static <- get_static_stack(id)

  # --- safety checks ---
  stopifnot(
    compareGeom(r_static, r_smap, stopOnError = FALSE)
  )
  
  # wc_bands <- c("WCpF2", "WCpF42", "WCres", "WCsat")
  # r_static[[wc_bands]] <- r_static[[wc_bands]] / 10000
  
  # Stack (static + dynamic)
  r_stack <- c(r_smap, r_lst, r_static)

  # Output name
#   out_name <- str_replace(basename(smap_file), "\\.tif$", "_stack.tif")
  out_name <- basename(smap_file)
  out_file <- file.path(out_dir, out_name)

  # Write compressed multiband GeoTIFF
  writeRaster(
    r_stack,
    out_file,
    overwrite = TRUE,
    wopt = list(
      gdal = c(
        "COMPRESS=LZW",
        "TILED=YES"
      )
    )
  )

  # message("Saved: ", out_name)
  i <- i+1
}
