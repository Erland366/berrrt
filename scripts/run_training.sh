bash scripts/print_header.sh

python main.py mode=full modules=bert modules_name=bert

# python main.py mode=sanity_check modules.layer_start=0 modules.layer_end=11 modules.gate=sigmoid modules=berrrt_gate modules_name=berrrt_gate


# # run training berrrt
for ((layer=0; layer<=10; layer++)); do
  # Execute the print_header.sh script
  bash scripts/print_header.sh
  
  # Run main.py with the current layer
  python main.py mode=full modules.layer_start=$layer modules.layer_end=11 modules=berrrt modules_name=berrrt modules.aggregation=average
done

for ((layer=0; layer<=10; layer++)); do
  # Execute the print_header.sh script
  bash scripts/print_header.sh
  
  # Run main.py with the current layer
  python main.py mode=full modules.layer_start=$layer modules.layer_end=11 modules=berrrt modules_name=berrrt modules.aggregation=add
done

for ((layer=0; layer<=10; layer++)); do
  # Execute the print_header.sh script
  bash scripts/print_header.sh
  
  # Run main.py with the current layer
  python main.py mode=full modules.layer_start=$layer modules.layer_end=11 modules=berrrt modules_name=berrrt modules.aggregation=pool
done

for ((layer=0; layer<=10; layer++)); do
  # Execute the print_header.sh script
  bash scripts/print_header.sh
  
  # Run main.py with the current layer
  python main.py mode=full modules.layer_start=$layer modules.layer_end=11 modules=berrrt modules_name=berrrt modules.aggregation=concat
done

for ((layer=0; layer<=10; layer++)); do
  # Execute the print_header.sh script
  bash scripts/print_header.sh
  
  # Run main.py with the current layer
  python main.py mode=full modules.layer_start=$layer modules.layer_end=11 modules=berrrt modules_name=berrrt modules.aggregation=weighted_sum
done

for ((layer=0; layer<=10; layer++)); do
  # Execute the print_header.sh script
  bash scripts/print_header.sh
  
  # Run main.py with the current layer
  python main.py mode=full modules.layer_start=$layer modules.layer_end=11 modules=berrrt modules_name=berrrt modules.aggregation=attention
done