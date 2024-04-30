#!/bin/bash

# Lista de métodos y valores de nFeatures a probar
methods=("ORB" "SIFT" "AKAZE")
nfeatures_values=(100 1000 10000 20000)

# Ruta al script principal que quieres ejecutar
script_path="Features.py"  
show_image="0"

# Nombre del archivo de resultados
output_file="resultados2.txt"

# Iterar sobre cada combinación de método y nFeatures
for method in "${methods[@]}"; do
    for nfeatures in "${nfeatures_values[@]}"; do
        # Ejecutar el script principal con el método y nFeatures actuales
        echo "Running script with method: $method, nFeatures: $nfeatures"
        echo "=====================================" >> "$output_file"
        echo "Method: $method, nFeatures: $nfeatures" >> "$output_file"
        python "$script_path" "$method" "$nfeatures" "$show_image" >> "$output_file"
        echo "-------------------------------------" >> "$output_file"
    done
done

echo "All tests completed. Results saved in $output_file"