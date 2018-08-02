#!/usr/bin/env bash





for network in alexnet inception_resnet_v2 densenet
do
    for embedding in 128 256 512
    do
        for data in DETAYGRUP_ADI ANAGRUP_ADI MARKA_ADI URUNBELBILGISI_ADI URUNBOYBILGISI_ADI URUNBURUNBILGISI_ADI URUNCEPBILGISI_ADI \
                URUNCINSIYET_ADI URUNDESENBILGISI_ADI URUNEBATBILGISI_ADI URUNKAPAMADETAYI_ADI URUNKIYAFETKOD_ADI URUNKOLBILGISI_ADI \
                URUNOKCEBILGISI_ADI URUNPACABILGISI_ADI URUNYAKABILGISI_ADI USTGRUP_ADI
        do
            python trainer.py --network $network --embedding $embedding --epochs 500 --batch_size 512 --feature_name $data

            python create_batches.py --network $network --embedding $embedding --batch_size 512 --feature_name $data
        done
    done
done
