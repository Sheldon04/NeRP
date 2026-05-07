for DATASET in eurosat dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft caltech101 food101 stanford_cars sun397 imagenet
do
    bash scripts/mmrl_nerp/base2new_test.sh $DATASET
done

# for DATASET in eurosat dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft caltech101 food101 stanford_cars sun397 imagenet
# do
#     bash scripts/mmadapter_nerp/base2new_test.sh $DATASET
# done

# for DATASET in eurosat dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft caltech101 food101 stanford_cars sun397 imagenet
# do
#     bash scripts/promptsrc_nerp/base2new_test.sh $DATASET
# done

# for DATASET in eurosat dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft caltech101 food101 stanford_cars sun397 imagenet
# do
#     bash scripts/cocoop_nerp/base2new_test.sh $DATASET
# done