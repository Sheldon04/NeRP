for DATASET in eurosat dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft caltech101 food101 stanford_cars sun397 imagenet
do
    bash scripts/mmrl/base2new_train.sh $DATASET
    bash scripts/mmrl/base2new_test.sh $DATASET
done

# for DATASET in eurosat dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft caltech101 food101 stanford_cars sun397 imagenet
# do
#     bash scripts/mmadapter/base2new_train.sh $DATASET
#     bash scripts/mmadapter/base2new_test.sh $DATASET
# done

# for DATASET in eurosat dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft caltech101 food101 stanford_cars sun397 imagenet
# do
#     bash scripts/promptsrc/base2new_train.sh $DATASET
#     bash scripts/promptsrc/base2new_test.sh $DATASET
# done

# for DATASET in eurosat dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft caltech101 food101 stanford_cars sun397 imagenet
# do
#     bash scripts/cocoop/base2new_train.sh $DATASET
#     bash scripts/cocoop/base2new_test.sh $DATASET
# done