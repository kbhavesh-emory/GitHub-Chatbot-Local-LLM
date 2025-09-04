#!/usr/bin/env bash
# clone_dsa_repos.sh - Enhanced with error handling and progress
set -euo pipefail

TARGET_DIR="backend/data"
LOG_FILE="clone_log_$(date +%Y%m%d_%H%M%S).txt"
FAILED_FILE="failed_repos.txt"

# Extensive repository list (same as before)
REPOS=(
  "https://github.com/pitt-bdsa/Dash-PaperDragon"
  "https://github.com/pitt-bdsa/bdsa"
  "https://github.com/pitt-bdsa/webapps"
  "https://github.com/digitalslidearchive/digital_slide_archive"
  "https://github.com/DigitalSlideArchive/HistomicsTK"
  "https://github.com/DigitalSlideArchive/ansible-role-vips"
  "https://github.com/DigitalSlideArchive/base_docker_image"
  "https://github.com/DigitalSlideArchive/dsa_girder_webix_base_viewer"
  "https://github.com/DigitalSlideArchive/DSA_Documentation"
  "https://github.com/DigitalSlideArchive/digitalslidearchive.info"
  "https://github.com/DigitalSlideArchive/pylibtiff"
  "https://github.com/DigitalSlideArchive/CNNCellDetection"
  "https://github.com/DigitalSlideArchive/ctk-cli"
  "https://github.com/DigitalSlideArchive/HistomicsUI"
  "https://github.com/DigitalSlideArchive/DSA-WSI-DeID"
  "https://github.com/DigitalSlideArchive/HistomicsStream"
  "https://github.com/DigitalSlideArchive/tifftools"
  "https://github.com/DigitalSlideArchive/annotation-tracker"
  "https://github.com/DigitalSlideArchive/ImageDePHI-Phase-I"
  "https://github.com/DigitalSlideArchive/HTAN"
  "https://github.com/DigitalSlideArchive/HistomicsDetect"
  "https://github.com/DigitalSlideArchive/girder-client-mount"
  "https://github.com/DigitalSlideArchive/ALBench"
  "https://github.com/DigitalSlideArchive/import-tracker"
  "https://github.com/DigitalSlideArchive/large-image-utilities"
  "https://github.com/DigitalSlideArchive/superpixel-classification"
  "https://github.com/DigitalSlideArchive/histoqc-dsa-plugin"
  "https://github.com/DigitalSlideArchive/wsi-superpixel-guided-labeling"
  "https://github.com/DigitalSlideArchive/ImageDePHI"
  "https://github.com/DigitalSlideArchive/large_image_source_isyntax"
  "https://github.com/DigitalSlideArchive/dive-dsa"
  "https://github.com/DigitalSlideArchive/girder_volview"
  "https://github.com/DigitalSlideArchive/histomics_load_testing"
  "https://github.com/DigitalSlideArchive/histomics-tour"
  "https://github.com/DigitalSlideArchive/dsa-run-custom-ai-models"
  "https://github.com/DigitalSlideArchive/girder-clamav"
  "https://github.com/DigitalSlideArchive/histomicstk-extras"
  "https://github.com/DigitalSlideArchive/girder_assetstore"
  "https://github.com/Gutman-Lab/WSVV"
  "https://github.com/Gutman-Lab/PML"
  "https://github.com/Gutman-Lab/bdsa-model-registry"
  "https://github.com/Gutman-Lab/ihc-tissue-detection"
  "https://github.com/Gutman-Lab/DinoV2_hf"
  "https://github.com/Gutman-Lab/WSIA"
  "https://github.com/Gutman-Lab/neurotk"
  "https://github.com/Gutman-Lab/WSU"
  "https://github.com/Gutman-Lab/dsa-helpers"
  "https://github.com/Gutman-Lab/neurotk-react"
  "https://github.com/Gutman-Lab/DSARequests"
  "https://github.com/Gutman-Lab/slicer-cli-example"
  "https://github.com/Gutman-Lab/raygun"
  "https://github.com/Gutman-Lab/dsa-emory-adrc"
  "https://github.com/Gutman-Lab/plaque-detection-yolo12"
  "https://github.com/Gutman-Lab/wsi-tissue-detection"
  "https://github.com/Gutman-Lab/dsa-czi-converter"
  "https://github.com/Gutman-Lab/WSIVDB"
  "https://github.com/Gutman-Lab/emory-path-qc"
  "https://github.com/Gutman-Lab/BlueBird"
  "https://github.com/Gutman-Lab/LargeImageIterator"
  "https://github.com/Gutman-Lab/comparing-tissue-detection-models"
  "https://github.com/Gutman-Lab/DeidTools"
  "https://github.com/Gutman-Lab/bdsa-workflows-slurm"
  "https://github.com/Gutman-Lab/WSIFE"
  "https://github.com/Gutman-Lab/tissue-plate-specimen-qc"
  "https://github.com/Gutman-Lab/WSIQC"
  "https://github.com/Gutman-Lab/PFM"
  "https://github.com/Gutman-Lab/abeta-tissue-compartment-detection"
  "https://github.com/Gutman-Lab/bdsa-workflows"
  "https://github.com/Gutman-Lab/emory-qc-viz"
  "https://github.com/Gutman-Lab/DSA-Annotation-Browser"
  "https://github.com/Gutman-Lab/ADRC-np-survey-2023-private"
  "https://github.com/Gutman-Lab/osd-paperjs-annotation"
  "https://github.com/Gutman-Lab/bdsa"
  "https://github.com/Gutman-Lab/DSA-Tissue-Reg"
  "https://github.com/Gutman-Lab/Positive-Pixel-Count-"
  "https://github.com/Gutman-Lab/nft-detection"
  "https://github.com/Gutman-Lab/triton-inference"
  "https://github.com/Gutman-Lab/DSA-LLM-Docs"
  "https://github.com/Gutman-Lab/yolov8-nft"
  "https://github.com/Gutman-Lab/cell-tracking-application"
  "https://github.com/Gutman-Lab/ppc-slurm"
  "https://github.com/Gutman-Lab/bdsa-readthedocs"
  "https://github.com/Gutman-Lab/yolo-braak-stage"
  "https://github.com/Gutman-Lab/BrainDigitalSlideArchive"
  "https://github.com/Gutman-Lab/tiling-wsis"
  "https://github.com/Gutman-Lab/wsi-schema-frontend"
  "https://github.com/Gutman-Lab/neuropath-foundation-model"
  "https://github.com/Gutman-Lab/BDSA-Schema-Wrangler"
  "https://github.com/Gutman-Lab/abeta-detection-project"
  "https://github.com/Gutman-Lab/ppc-profiler"
  "https://github.com/Gutman-Lab/BrainSec-py"
  "https://github.com/Gutman-Lab/tissue-detection-att-unet"
  "https://github.com/Gutman-Lab/bdsarepochat"
  "https://github.com/Gutman-Lab/levey-project"
  "https://github.com/Gutman-Lab/dsa-slicer-cli-web-tasks"
  "https://github.com/Gutman-Lab/Dash-Plotly-YOLO"
  "https://github.com/Gutman-Lab/DSAClusterExploration"
  "https://github.com/Gutman-Lab/emory-adrc-dsa"
  "https://github.com/Gutman-Lab/arjita-gray-matter-detection"
  "https://github.com/Gutman-Lab/ADRC-np-survey-2023"
)


# Convert to SSH format
to_ssh() {
    local url="$1"
    url="${url%/}"
    url="${url%%.git}"
    if [[ "$url" =~ ^https?://(www\.)?github\.com/([^/]+)/([^/]+)$ ]]; then
        echo "git@github.com:${BASH_REMATCH[2]}/${BASH_REMATCH[3]}.git"
    elif [[ "$url" =~ ^git@github\.com:([^/]+)/([^/]+)(\.git)?$ ]]; then
        echo "git@github.com:${BASH_REMATCH[1]}/${BASH_REMATCH[2]}.git"
    else
        echo "$1"
    fi
}

# Initialize
mkdir -p "$TARGET_DIR"
echo "📋 Repository cloning started at $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

fail_count=0
fail_list=()
success_count=0

for repo in "${REPOS[@]}"; do
    ssh_url="$(to_ssh "$repo")"
    name="$(basename "${ssh_url%%.git}")"
    dir="${TARGET_DIR}/${name}"
    
    echo "🔄 Processing: $name" | tee -a "$LOG_FILE"
    
    if [[ -d "$dir/.git" ]]; then
        echo "   ↻ Updating existing repository..." | tee -a "$LOG_FILE"
        if git -C "$dir" fetch --all --prune && git -C "$dir" pull --ff-only; then
            echo "   ✅ Update successful" | tee -a "$LOG_FILE"
            ((success_count++))
        else
            echo "   ❌ Update failed" | tee -a "$LOG_FILE"
            fail_list+=("$name (update)")
            ((fail_count++))
            echo "$name" >> "$FAILED_FILE"
        fi
    else
        echo "   📥 Cloning new repository..." | tee -a "$LOG_FILE"
        if git clone --depth 1 --progress "$ssh_url" "$dir" 2>&1 | tee -a "$LOG_FILE"; then
            echo "   ✅ Clone successful" | tee -a "$LOG_FILE"
            ((success_count++))
        else
            echo "   ❌ Clone failed" | tee -a "$LOG_FILE"
            fail_list+=("$name (clone)")
            ((fail_count++))
            echo "$name" >> "$FAILED_FILE"
        fi
    fi
    echo "---" | tee -a "$LOG_FILE"
done

# Summary
echo "==========================================" | tee -a "$LOG_FILE"
echo "📊 CLONING SUMMARY" | tee -a "$LOG_FILE"
echo "✅ Successful: $success_count" | tee -a "$LOG_FILE"
echo "❌ Failed: $fail_count" | tee -a "$LOG_FILE"

if (( fail_count > 0 )); then
    echo "⚠️ Failed repositories:" | tee -a "$LOG_FILE"
    printf '   - %s\n' "${fail_list[@]}" | tee -a "$LOG_FILE"
    echo "   Details saved to: $FAILED_FILE" | tee -a "$LOG_FILE"
fi

echo "📋 Full log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "🎉 Cloning completed at $(date)" | tee -a "$LOG_FILE"