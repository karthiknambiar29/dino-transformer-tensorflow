import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import os
import random
from PIL import Image
import matplotlib.pyplot as plt

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def img_to_array(image_path):
    x = np.asarray(Image.open(image_path))
    x = tf.image.resize(x, (480, 480), method='bicubic').numpy()
    x = tf.image.per_image_standardization(x).numpy()
    return x

def plot_attention(model, image_paths, patch_size, print_head=0, save_fig=False):
    image_arrays = np.array([img_to_array(image_path) for image_path in image_paths])
    w_featmap, h_featmap = image_arrays[0].shape[1] // patch_size, image_arrays[0].shape[1] // patch_size
    attentions = model.get_last_selfattention(image_arrays).numpy()
    batch_size, nh = attentions.shape[0], attentions.shape[1]
    attentions = attentions[:, :, 0, 1:].reshape(batch_size, nh, -1)
    attentions = attentions.reshape(batch_size, nh, w_featmap, h_featmap)
    attentions = np.transpose(attentions, (0, 3, 2, 1))
    attentions = tf.image.resize(attentions, (480, 480), method='nearest').numpy()
    if type(print_head) != str:
        attentions = attentions[:, :, :, print_head]
        for i in range(batch_size):
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(image_arrays[i])
            ax1.set_title('image')
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax2.imshow(attentions[i])
            ax2.set_title('attention')
            ax2.set_xticks([])
            ax2.set_yticks([])
            if save_fig:
                plt.savefig(str(i)+'.png')
            plt.show()
    if print_head == 'average':
        attentions = attentions.mean(axis=-1)
        for i in range(batch_size):
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(image_arrays[i])
            ax1.set_title('image')
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax2.imshow(attentions[i])
            ax2.set_title('mean attention')
            ax2.set_xticks([])
            ax2.set_yticks([])
            if save_fig:
                plt.savefig(str(i)+'.png')
            plt.show()
    if print_head == 'all':
        for i in range(batch_size):
            fig = plt.figure(figsize=(5*(nh+1), 5))
            #print(nh)
            for j in range(0, nh+1):
                ax = fig.add_subplot(1, nh+1, j+1)
                if j==0:
                    ax.imshow(image_arrays[i])
                    ax.set_title('image')
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    ax.imshow(attentions[i, :, :, j-1])
                    ax.set_title(str(j)+' attention head')
                    ax.set_xticks([])
                    ax.set_yticks([])
            if save_fig:
                plt.savefig(str(i)+'.png')
            plt.show()
    return attentions

class DataAugmentationDINO():
    def __init__(self, global_crop_scale, local_crop_scale, local_crops_number, image_shape=(224, 224, 3)):

        self.global_crop_scale = global_crop_scale
        self.local_crop_scale = local_crop_scale
        self.local_crops_number = local_crops_number
        self.img_height = image_shape[0]
        self.img_width = image_shape[1]
        self.in_chans = image_shape[2]
        
    def global_transform(self, img):
        out = []
        for x in img:
            x = tf.image.random_crop(x, size=(np.int32(self.img_height*self.global_crop_scale[0]),
                                              np.int32(self.img_width*self.global_crop_scale[1]), self.in_chans))
            x = tf.image.resize(x, (224, 224), method='bicubic').numpy()
            out.append(x)
        return np.array(out)
    
    def local_transform(self, img):
        out = []
        for x in img:
            x = tf.image.random_crop(x, size=(np.int32(self.img_height*self.global_crop_scale[0]),
                                              np.int32(self.img_width*self.global_crop_scale[1]), self.in_chans))
            x = tf.image.resize(x, (224, 224), method='bicubic').numpy()
            out.append(x)
        return np.array(out)
        
    def augment(self, img):
        out = []
        for x in img:
            x = tf.image.random_flip_left_right(x)
            x = tf.image.random_contrast(x, 0.09, 1.01).numpy()
            out.append(x)
        return np.array(out)
        
        
    def __call__(self, images):
        crop = []
        x = self.augment(images)
        crop.append(self.global_transform(x).astype(int))
        crop.append(self.global_transform(x).astype(int))
        for _ in range(self.local_crops_number):
            crop.append(self.local_transform(x).astype(int))
        return tuple(crop)

class DINOLoss(tf.keras.losses.Loss):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                center_momentum=0.9):
        super(DINOLoss, self).__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.center = tf.Variable(tf.zeros((1, out_dim)), trainable=False, name="center")
        
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        
    def call(self, student_output, teacher_output, epoch):
        student_out = student_output / self.student_temp
        student_out = tf.split(student_out, self.ncrops)

        temp = self.teacher_temp_schedule[epoch]
        teacher_out = tf.nn.softmax((teacher_output - self.center)/temp, axis=-1)

        
        teacher_out = tf.split(tf.stop_gradient(teacher_out), 2)

        total_loss = 0.0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v==iq:
                    continue
                loss = tf.math.reduce_sum(-q * tf.nn.log_softmax(student_out[v], axis=-1), axis=-1)
                total_loss += tf.math.reduce_mean(loss)
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    def update_center(self, teacher_output):
        batch_center = tf.stop_gradient(tf.math.reduce_sum(teacher_output, axis=0, keepdims=True))
        batch_center = batch_center / len(teacher_output)

        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class MultiCropWrapper(tf.keras.Model):
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        backbone.head = tf.keras.layers.Lambda(lambda x:x)
        self.backbone = backbone
        self.head = head
        
    def call(self, x):
        idx_crops = tf.math.cumsum(tf.unique_with_counts([inp.shape[1] for inp in x])[-1],
                                   axis=0)

        start_idx, output = 0, []
        for end_idx in idx_crops:
            _out, _ = self.backbone(np.concatenate(x[start_idx: end_idx], axis=0))

            if isinstance(_out, tuple):
                _out = _out[0]
            output.append(_out)
            start_idx = end_idx
        temp = output[0]
        for i in range(1, len(output)):
            temp = np.concatenate([temp, output[i]])
        output = temp
        return self.head(output)

class datagen():
    def __init__(self, dir_path, batch_size=1, image_shape=(224, 224), shuffle=True):
        self.dir_path = dir_path
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.class_names = os.listdir(dir_path)
        self.file_paths = []
        self.labels = []
        self.n = 0
        for i, names in enumerate(self.class_names):
            paths = os.listdir(os.path.join(self.dir_path, names))
            for path in paths:
                img = np.asarray(Image.open(os.path.join(self.dir_path, names, path)).resize(self.image_shape))
                if img.shape == (224, 224, 3):
                    self.labels.append(i)
                    self.file_paths.append(os.path.join(self.dir_path, names, path))
        self.max = self.num_batch = len(self.labels)
        if shuffle:
            temp = list(zip(self.labels, self.file_paths))
            random.shuffle(temp)
            self.labels, self.file_paths = zip(*temp)
            self.labels, self.file_paths = list(self.labels), list(self.file_paths)
        self.file_paths = [self.file_paths[i:i+self.batch_size] for i in range(0, len(self.file_paths), self.batch_size)]
        self.labels = [self.labels[i:i+self.batch_size] for i in range(0, len(self.labels), self.batch_size)]
        self.num_batch = int(np.ceil(self.max / self.batch_size))
        
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.n == self.num_batch:
            self.n = 0
            raise StopIteration
        
        if self.n < self.num_batch:
            result = []
            for path in self.file_paths[self.n]:
                img = np.asarray(Image.open(path).resize(self.image_shape))
                #if img.shape == (224, 224, 3):
                result.append(img)
        self.n += 1
        return np.array(result)          
        
    def __len__(self):
        return self.num_batch


class Schedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    
    def __init__(self,base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
        super().__init__()
        warmup_schedule = np.array([])
        warmup_iters = warmup_epochs * niter_per_ep
        if warmup_epochs > 0:
            warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

        schedule = np.concatenate((warmup_schedule, schedule))
        assert len(schedule) == epochs * niter_per_ep
        self.schedule = schedule
        
    def __call__(self, step):
        return self.schedule[step]

def get_params_groups(model):
    regularized = []
    not_regularized = []
    for weights in model.trainable_weights:
        if "bias" in weights.name or len(weights.shape) == 1:
            not_regularized.append(weights)
        else:
            regularized.append(weights)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.0}]
