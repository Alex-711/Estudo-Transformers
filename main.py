from braindecode import EEGClassifier
from sklearn.model_selection import train_test_split

from models import BENDR
from braindecode.datasets import SleepPhysionet
from braindecode.preprocessing import preprocess, Preprocessor
from sklearn.preprocessing import robust_scale
from braindecode.preprocessing import create_windows_from_events
import numpy as np
from sklearn.utils import compute_class_weight
from skorch.helper import predefined_split
from skorch.callbacks import EpochScoring
from sklearn.metrics import balanced_accuracy_score
import torch
from braindecode.util import set_random_seeds
from braindecode.samplers import SequenceSampler




if __name__ == "__main__":

    subject_ids = [0, 1]
    crop = (0, 30 * 400)
    dataset = SleepPhysionet(
        subject_ids=subject_ids, recording_ids=[2], crop_wake_mins=30)

    preprocessors = [Preprocessor(robust_scale, channel_wise=True)]

    preprocess(dataset, preprocessors)

    mapping = {  # We merge stages 3 and 4 following AASM standards.
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3': 3,
        'Sleep stage 4': 3,
        'Sleep stage R': 4,
    }

    window_size_s = 30
    sfreq = 100
    window_size_samples = window_size_s * sfreq

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples,
        preload=True,
        mapping=mapping,
    )
    subject_ids= np.unique(windows_dataset.description['subject'])
    crop = (0, 30 * 400)
    dataset = SleepPhysionet(
        subject_ids=subject_ids, recording_ids=[2], crop_wake_mins=30)



    preprocessors = [Preprocessor(robust_scale, channel_wise=True)]

    preprocess(dataset, preprocessors)



    mapping = {  # We merge stages 3 and 4 following AASM standards.
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3': 3,
        'Sleep stage 4': 3,
        'Sleep stage R': 4,
    }

    window_size_s = 30
    sfreq = 100
    window_size_samples = window_size_s * sfreq

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples,
        preload=True,
        mapping=mapping,
    )
    split_ids = dict(train=subject_ids[::2], valid=subject_ids[1::2])
    train_valid_ix, test_ix = split_ids["train"], split_ids["valid"]

    splits = windows_dataset.split(split_ids)
    train_set, valid_set = splits["train"], splits["valid"]

    df = windows_dataset.description
    train_ix, valid_ix = train_test_split(train_set, train_size=0.8)
    train_ix = df[df['subject'].isin(train_ix)].index.tolist()
    valid_ix = df[df['subject'].isin(valid_ix)].index.tolist()
    test_ix = df[df['subject'].isin(test_ix)].index.tolist()

    y_true_train = train_set.get_metadata()['target'].to_numpy()
    class_weights = compute_class_weight('balanced', classes=np.unique(y_true_train), y=y_true_train)
    model_hparams = {
        'model': 'BENDR',
        'encoder_h': 512,
        'projection_head': False,
        'enc_do': 0.1,
        'feat_do': 0.4,
        'pool_length': 4,
        'mask_p_t': 0.01,
        'mask_p_c': 0.005,
        'mask_t_span': 0.05,
        'mask_c_span': 0.1,
        'classifier_layers': 1,
        'model_path': None
    }

    model = BENDR(windows_input=1000, samples=3000, original_channel_size=4, model_hparams=model_hparams)



    lr = 1e-3
    batch_size = 32
    n_epochs = 3


    cuda = torch.cuda.is_available()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if cuda:
        torch.backends.cudnn.benchmark = True

    set_random_seeds(seed=31, cuda=cuda)

    n_classes = 5

    in_chans, input_size_samples = train_set[0][0].shape


    def balanced_accuracy_multi(model, X, y):
        y_pred = model.predict(X)
        return balanced_accuracy_score(y.flatten(), y_pred.flatten())


    train_bal_acc = EpochScoring(
        scoring=balanced_accuracy_multi,
        on_train=True,
        name='train_bal_acc',
        lower_is_better=False,
    )
    valid_bal_acc = EpochScoring(
        scoring=balanced_accuracy_multi,
        on_train=False,
        name='valid_bal_acc',
        lower_is_better=False,
    )
    callbacks = [
        ('train_bal_acc', train_bal_acc),
        ('valid_bal_acc', valid_bal_acc)
    ]

    clf = EEGClassifier(
        model,
        criterion=torch.nn.CrossEntropyLoss,
        criterion__weight=torch.Tensor(class_weights).to(device),
        optimizer=torch.optim.Adam,
        iterator_train__shuffle=False,
        iterator_train__sampler=train_ix,
        iterator_valid__sampler=valid_ix,
        train_split=predefined_split(valid_set),
        optimizer__lr=lr,
        batch_size=batch_size,
        callbacks=callbacks,
        device=device,
    )
    clf.fit(train_set, y=None, epochs=n_epochs)

