
## "emnist-balanced" dataset (original 243mb compressed to 31mb)

The EMNIST Balanced dataset is meant to address the balance issues in the ByClass and ByMerge datasets. It is derived from the ByMerge dataset to reduce mis-classification errors due to capital and lower case letters and also has an equal number of samples per class. This dataset is meant to be the most applicable.

    train: 112,800
    test: 18,800
    total: 131,600
    classes: 47 (balanced)
    
## "emnist-bymerge" dataset (original 1.46gb compressed to 0mb)
The full complement of the NIST Special Database 19 is available in the ByClass and ByMerge splits. These two datasets have the same image information but differ in the number of images in each class. Both datasets have an uneven number of images per class and there are more digits than letters. The number of letters roughly equate to the frequency of use in the English language.

    train: 697,932
    test: 116,323
    total: 814,255
    classes: 47 (unbalanced)

## "emnist-byclass" dataset (original 1.46gb compressed to 188mb)
The full complement of the NIST Special Database 19 is available in the ByClass and ByMerge splits. These two datasets have the same image information but differ in the number of images in each class. Both datasets have an uneven number of images per class and there are more digits than letters. The number of letters roughly equate to the frequency of use in the English language.

    train: 697,932
    test: 116,323
    total: 814,255
    classes: 62 (unbalanced)

## "emnist-letters" dataset (original 190mb compressed to 24mb)
The EMNIST Letters dataset merges a balanced set of the uppercase and lowercase letters into a single 26-class task.

    train: 88,800
    test: 14,800
    total: 103,600
    classes: 37 (balanced)

## "emnist-mnist" dataset (original: 129mb compressed to 16mb)
The EMNIST MNIST dataset provide balanced handwritten digit datasets directly compatible with the original MNIST dataset.

    train: MNIST 60,000
    test: MNIST 10,000
    total: MNIST 70,000
    classes: 10 (balanced)

## "emnist-digits" dataset (original 516mb compressed to 64mb)
The EMNIST Digits dataset provide balanced handwritten digit datasets directly compatible with the original MNIST dataset.

    train: Digits 240,000
    test: Digits 40,000
    total: Digits 280,000
    classes: 10 (balanced)
