bash scripts/print_header.sh

source .env

datasets=("emotion" "hate" "irony" "rte")

for dataset in "${datasets[@]}"; do
  python main.py mode=full modules=bert modules_name=bert dataset=$dataset

  # # run training berrrt
  for ((layer=0; layer<=10; layer++)); do
    # Execute the print_header.sh script
    bash scripts/print_header.sh
    
    # Run main.py with the current layer
    python main.py mode=full modules.layer_start=$layer modules.layer_end=11 modules=berrrt modules_name=berrrt modules.aggregation=average dataset=$dataset

  done

  for ((layer=0; layer<=10; layer++)); do
    # Execute the print_header.sh script
    bash scripts/print_header.sh
    
    # Run main.py with the current layer
    python main.py mode=full modules.layer_start=$layer modules.layer_end=11 modules=berrrt modules_name=berrrt modules.aggregation=add dataset=$dataset

  done

  for ((layer=0; layer<=10; layer++)); do
    # Execute the print_header.sh script
    bash scripts/print_header.sh
    
    # Run main.py with the current layer
    python main.py mode=full modules.layer_start=$layer modules.layer_end=11 modules=berrrt modules_name=berrrt modules.aggregation=concat dataset=$dataset

  done

  for ((layer=0; layer<=10; layer++)); do
    # Execute the print_header.sh script
    bash scripts/print_header.sh
    
    # Run main.py with the current layer
    python main.py mode=full modules.layer_start=$layer modules.layer_end=11 modules=berrrt modules_name=berrrt modules.aggregation=weighted_sum dataset=$dataset

  done

  for ((layer=0; layer<=10; layer++)); do
    # Execute the print_header.sh script
    bash scripts/print_header.sh
    
    # Run main.py with the current layer
    python main.py mode=full modules.layer_start=$layer modules.layer_end=11 modules=berrrt modules_name=berrrt modules.aggregation=attention dataset=$dataset

  done

  for ((layer=0; layer<=10; layer++)); do
    # Execute the print_header.sh script
    bash scripts/print_header.sh
    
    # Run main.py with the current layer
    python main.py mode=full modules.layer_start=$layer modules.layer_end=11 modules=berrrt_gate modules_name=berrrt_gate modules.gate=attention dataset=$dataset

  done

  for ((layer=0; layer<=10; layer++)); do
    # Execute the print_header.sh script
    bash scripts/print_header.sh
    
    # Run main.py with the current layer
    python main.py mode=full modules.layer_start=$layer modules.layer_end=11 modules=berrrt_gate modules_name=berrrt_gate modules.gate=softmax dataset=$dataset

  done

  for ((layer=0; layer<=10; layer++)); do
    # Execute the print_header.sh script
    bash scripts/print_header.sh
    
    # Run main.py with the current layer
    python main.py mode=full modules.layer_start=$layer modules.layer_end=11 modules=berrrt_gate modules_name=berrrt_gate modules.gate=sigmoid dataset=$dataset

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