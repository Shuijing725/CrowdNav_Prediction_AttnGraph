mkdir -p datasets/wrapper_demo
cd datasets/wrapper_demo
# use wget for small-sized google drive files
wget -O wrapper_demo_data.pt "https://drive.google.com/uc?export=download&id=1Fjc4nFAeqgCTd9UEUsXhJkESqgErNtlA" > /tmp/intermezzo.html

cd ../../results
# use wget for small-sized google drive files
wget -O 211222-model.zip "https://drive.google.com/uc?export=download&id=1mm364PGbp7hFHJ8uXN_uZI4Nq_3JIHPj" > /tmp/intermezzo.html
unzip 211222-model.zip && rm -rf 211222-model.zip