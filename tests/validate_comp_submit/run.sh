python av2_scene_flow_competition_submit.py /tmp/argoverse2_tiny/val_sceneflow_feather/ --validation_json tests/validate_comp_submit/tiny_val_sceneflow_feather_counts.json --is_supervised False

# Unzip output /tmp/argoverse2_tiny/val_sceneflow_feather_av2_2024_sf_submission.zip to /tmp/comp_output
rm -rf /tmp/comp_output
unzip -q /tmp/argoverse2_tiny/val_sceneflow_feather_av2_2024_sf_submission.zip -d /tmp/comp_output

# Validate there is a single folder /tmp/comp_output/02678d04-cc9f-3148-9f95-1ba66347dff9/
if [ ! -d /tmp/comp_output/02678d04-cc9f-3148-9f95-1ba66347dff9/ ]; then
  echo "Expected folder /tmp/comp_output/02678d04-cc9f-3148-9f95-1ba66347dff9/ not found"
  exit 1
fi

# Validate there is a single feather file inside this folder
if [ ! -f /tmp/comp_output/02678d04-cc9f-3148-9f95-1ba66347dff9/*.feather ]; then
  echo "Expected file /tmp/comp_output/02678d04-cc9f-3148-9f95-1ba66347dff9/*.feather not found"
  exit 1
fi
