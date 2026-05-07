import re
from collections import defaultdict
import os
import requests

path_to_metadata = "D:\SM-DeepLearning\datasets\SMAP\\5000006492388.txt"
save_path = "D:\SM-DeepLearning\datasets\SMAP\\5000006492388_links.txt"
data_path = "D:\SM-DeepLearning\datasets\SMAP\\2015_19\\"

failed_path = "D:\SM-DeepLearning\SMP25-SM-ML2\\failed_downloads.txt"

def smap_links(input_file: str, output_file: str):
    """
    Extracts core 'soil_moisture' .tif URLs (excluding variants like _dca, _error, _scah, _scav),
    writes them to an output file, and returns the matching count and dates.

    Args:
        input_file (str): Path to the input metadata file.
        output_file (str): Path to the output file for filtered URLs.

    Returns:
        tuple: (count of links, dict of {date (str): [url1, url2, ...]})
    """
    pattern = re.compile(r".*/SMAP_L3_SM_P_(\d{8})_.*_soil_moisture_[^_/]*\.tif$")
    date_to_urls = defaultdict(list)
    count = 0

    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            url = line.strip()
            match = pattern.search(url)
            if match and not any(x in url for x in ["_dca", "_scah", "_scav", "_error"]):
                date = match.group(1)
                date_to_urls[date].append(url)
                outfile.write(url + "\n")
                count += 1

    print(f"{count} core soil moisture links written to {output_file}")
    return count, dict(date_to_urls)


import os
import requests

def download_soil_moisture_files(link_file: str, download_dir: str, failed_log: str = "failed_downloads.txt"):
    """
    Downloads .tif files from a list of URLs one by one, with progress tracking and error logging.

    Args:
        link_file (str): Path to the file containing URLs (one per line).
        download_dir (str): Directory to save the downloaded .tif files.
        failed_log (str): File to store failed URLs.

    Returns:
        int: Number of successfully downloaded files.
    """
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # Read all URLs
    with open(link_file, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    total = len(urls)
    success_count = 0
    failed_links = []

    for idx, url in enumerate(urls, start=1):
        filename = os.path.basename(url)
        filepath = os.path.join(download_dir, filename)

        print(f"[{idx}/{total}] Downloading: {filename}")

        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            with open(filepath, "wb") as out_file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        out_file.write(chunk)

            print(f"✓ Completed: {filename}")
            success_count += 1

        except requests.RequestException as e:
            print(f"✗ Failed: {filename} | Reason: {e}")
            failed_links.append(url)

    # Write failed links to file
    if failed_links:
        with open(failed_log, "w") as fail_file:
            for bad_url in failed_links:
                fail_file.write(bad_url + "\n")
        print(f"\n{len(failed_links)} failed downloads logged in: {failed_log}")

    print(f"\nAll downloads finished. Success: {success_count}, Failed: {len(failed_links)}")
    return success_count


# count, date_links = smap_links(path_to_metadata, save_path)

# # Print summary
# for date, urls in date_links.items():
#     print(f"{date}")
# print("Total:", count)

download_soil_moisture_files(failed_path, data_path)