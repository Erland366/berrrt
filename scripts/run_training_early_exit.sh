bash scripts/print_header.sh

source .env

datasets=("default" "emotion" "hate" "irony" "rte")

for dataset in "${datasets[@]}"; do
  python main.py mode=full modules=bert modules_name=bert dataset=$dataset

  # for ((layer=0; layer<=10; layer++)); do
  #   # Execute the print_header.sh script
  #   bash scripts/print_header.sh
    
  #   # Run main.py with the current layer
  #   python main.py mode=full modules.layer_start=$layer modules.layer_end=11 modules=berrrt_early_exit modules_name=berrrt_early_exit modules.gate=attention dataset=$dataset

  # done

  for ((layer=0; layer<=10; layer++)); do
    # Execute the print_header.sh script
    bash scripts/print_header.sh
    
    # Run main.py with the current layer
    python main.py mode=full modules.layer_start=$layer modules.layer_end=11 modules=berrrt_early_exit modules_name=berrrt_early_exit modules.gate=softmax dataset=$dataset

  done

  for ((layer=0; layer<=10; layer++)); do
    # Execute the print_header.sh script
    bash scripts/print_header.sh
    
    # Run main.py with the current layer
    python main.py mode=full modules.layer_start=$layer modules.layer_end=11 modules=berrrt_early_exit modules_name=berrrt_early_exit modules.gate=sigmoid dataset=$dataset

  done
done


if [ "$#" -eq 1 ]; then
    ID=$1
    API_KEY=$VASTAI_API_KEY

    curl --location -g --request DELETE "https://console.vast.ai/api/v0/instances/${ID}/" \
         --header "Accept: application/json" \
         --header "Authorization: Bearer ${API_KEY}"
else
    echo "ID not provided. Skipping delete vastai machine."
fi