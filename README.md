
## "balanced" dataset (original 243mb compressed to 31mb)

The EMNIST Balanced dataset is meant to address the balance issues in the ByClass and ByMerge datasets. It is derived from the ByMerge dataset to reduce mis-classification errors due to capital and lower case letters and also has an equal number of samples per class. This dataset is meant to be the most applicable.

    train: 112,800
    test: 18,800
    total: 131,600
    classes: 47 (balanced)
    
## "byClass dataset (original 1.46gb compressed to 0mb)
## "byMerge" dataset (original 1.46gb compressed to 0mb)
The full complement of the NIST Special Database 19 is available in the ByClass and ByMerge splits. These two datasets have the same image information but differ in the number of images in each class. Both datasets have an uneven number of images per class and there are more digits than letters. The number of letters roughly equate to the frequency of use in the English language.

    train: 697,932
    test: 116,323
    total: 814,255
    classes: ByClass 62 (unbalanced) / ByMerge 47 (unbalanced)

## "letters" dataset (original 190mb compressed to 0mb)
The EMNIST Letters dataset merges a balanced set of the uppercase and lowercase letters into a single 26-class task.

    train: 88,800
    test: 14,800
    total: 103,600
    classes: 37 (balanced)

## "mnist" dataset (original: 129mb compressed to 16mb)
## "digits" dataset (original 516mb compressed to 0mb)
The EMNIST Digits and EMNIST MNIST dataset provide balanced handwritten digit datasets directly compatible with the original MNIST dataset.

    train: Digits 240,000 / MNIST 60,000
    test: Digits 40,000 / MNIST 10,000
    total: Digits 280,000 / MNIST 70,000
    classes: 47 (balanced)
