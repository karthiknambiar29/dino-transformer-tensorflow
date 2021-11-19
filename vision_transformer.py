import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
import math

class droppath(tf.keras.layers.Layer):
    def __init__(self, drop_prob):
        super().__init__()
        # parameters
        self.drop_prob = drop_prob
        self.keep_prob = 1 - drop_prob
        
        # layers
        self.random_norm = tf.keras.layers.Lambda(lambda x:tf.random.normal(tf.shape(x)))
        self.add = tf.keras.layers.Lambda(lambda x:tf.math.add(x[0],x[1]))
        self.floor = tf.keras.layers.Lambda(lambda x:tf.math.floor(x))
        self.divide = tf.keras.layers.Lambda(lambda x : tf.math.divide(x[0], x[1]) * x[2])

    def call(self, x):
        inputs = x
        shape = inputs.shape[1:]
        x = self.random_norm(x)
        x = self.add([x, self.keep_prob])
        x = self.floor(x)
        x = self.divide([inputs, self.keep_prob, x])
        return x

class DropPath(tf.keras.layers.Layer):
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        self.drop_path = droppath(self.drop_prob)
        self.identity_layer = tf.keras.layers.Lambda(lambda x:x)

    def call(self, x):
        if self.drop_prob==0.0 or not self.drop_path.trainable:
            x = self.identity_layer(x)
            return x
        return self.drop_path(x)

class Mlp(tf.keras.layers.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, activation=None, dropout=0.0, **kwargs):
        super().__init__()
        #parameters
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dense_1 = tf.keras.layers.Dense(units=hidden_features, input_shape=(None, in_features), activation=activation)
        self.drop_1 = tf.keras.layers.Dropout(dropout)
        self.dense_2 = tf.keras.layers.Dense(units=out_features, activation=None)
        self.drop_2 = tf.keras.layers.Dropout(dropout)

    def call(self, x):
        x = self.dense_1(x)
        x = self.drop_1(x)
        x = self.dense_2(x)
        x = self.drop_2(x)
        return x

class PatchEmbed(tf.keras.layers.Layer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.kwargs = kwargs
        self.conv2d = tf.keras.layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, dtype=tf.float32)

    def call(self, x):
        x = tf.cast(x, dtype=tf.float32)
        self.num_patches = int((x.shape[1] // self.patch_size) ** 2)
        x = self.conv2d(x)

        x = tf.keras.layers.Permute((2, 1, 3))(x)
        #print(x.shape)
        x = tf.keras.layers.Reshape((self.num_patches, self.embed_dim))(x)

        return x

class Attention(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads, patch_size=16, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, **kwargs):
        super().__init__()
        head_dim = dim // num_heads
        scale = qk_scale or head_dim ** -0.5
        self.scale = scale
        self.head_dim = head_dim
        self.dim = dim
        self.num_heads = num_heads
        self.pathc_size = patch_size
        self.qkv_bias = qkv_bias
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.kwargs = kwargs
        self.dense_1 = tf.keras.layers.Dense(dim*3, use_bias=qkv_bias)
        self.softmax = tf.keras.layers.Softmax()
        self.drop_attn = tf.keras.layers.Dropout(attn_drop)
        self.dense_2 = tf.keras.layers.Dense(dim)
        self.drop_proj = tf.keras.layers.Dropout(proj_drop)
    def call(self, x):
        input_shape = x.shape[1]
        x = self.dense_1(x)
        x = tf.keras.layers.Reshape((input_shape,3,self.num_heads, self.dim//self.num_heads))(x)
        x = tf.transpose(x, perm=[2, 0, 3, 1, 4])
       
        q, k, v = tf.split(x, 3, 0)
        q = q[0]
        k = k[0]
        v = v[0]

        k = tf.keras.layers.Permute((1, 3, 2))(k)
        attn = tf.keras.layers.Lambda(lambda x : tf.linalg.matmul(x[0], x[1]) * self.scale)([q, k])
        attn = self.softmax(attn)
        attn = self.drop_attn(attn)
        v = tf.keras.layers.Lambda(lambda x : tf.linalg.matmul(x[0], x[1]))([attn, v])
        x = tf.keras.layers.Permute((2, 1, 3))(v)
        x = tf.keras.layers.Reshape((input_shape, self.dim))(x)
        x = self.dense_2(x)
        x = self.drop_proj(x)
        return x, attn

class Block(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads,
                 patch_size=16, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0.,
          drop_path=0., act_layer=tf.nn.gelu, **kwargs):
        super(Block, self).__init__()
        # paramters
        self.dim = dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop = drop
        self.attn_drop = attn_drop
        self.drop_path = drop_path
        self.act_layer = act_layer
        self.kwargs = kwargs
        self.mlp_hidden_dim = int(self.dim * self.mlp_ratio)

        # layers
        self.layer_norm_1 = tf.keras.layers.LayerNormalization()
        self.attention = Attention(self.dim, num_heads=self.num_heads,
                                   patch_size=self.patch_size, qkv_bias=self.qkv_bias,
                                   qk_scale=self.qk_scale,
                                   attn_drop=self.attn_drop, proj_drop=self.drop)
        self.drop_path_1 = DropPath(self.drop_path)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization()
        self.mlp = Mlp(in_features=self.dim, hidden_features=self.mlp_hidden_dim,
                activation=self.act_layer, dropout=self.drop)
        self.drop_path_2 = DropPath(self.drop_path)
    
    def call(self, x, return_attention=False):
        inputs = x
        x = self.layer_norm_1(x)
        x, attn = self.attention(x)
        if return_attention:
            return attn
        x = self.drop_path_1(x)
        x = tf.keras.layers.Add()([x, inputs])
        y = self.layer_norm_2(x)
        y = self.mlp(y)
        y = self.drop_path_2(y)
        x = tf.keras.layers.Add()([x, y])

        return x

class VisionTransformer(tf.keras.Model):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=2,
                     num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                     drop_path_rate=0., **kwargs):
        super(VisionTransformer, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = int((self.img_size // self.patch_size) ** 2)
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.head = tf.keras.layers.Dense(num_classes) if self.num_classes > 0 else tf.keras.layers.Lambda(lambda x : x)
        self.patch_embed = PatchEmbed(patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.embed_dim)
        self.cls_token = tf.Variable(tf.zeros((1, 1, self.embed_dim)), name='cls_token')
        self.concat = tf.keras.layers.Concatenate(axis=1)
        self.pos_embed = tf.Variable(tf.zeros((1, self.num_patches+1, self.embed_dim)), name='pos_embed')
        self.layer_add = tf.keras.layers.Add()
        self.pos_drop = tf.keras.layers.Dropout(self.drop_rate)
        self.dpr = [elem.numpy() for elem in tf.linspace(0.0, self.drop_path_rate, self.depth)]
        self.blocks = [Block(dim=self.embed_dim,
                             num_heads=self.num_heads,
                             patch_size=self.patch_size,
                             mlp_ratio=self.mlp_ratio,
                             qkv_bias = self.qkv_bias,
                             qk_scale=self.qk_scale, drop=self.drop_rate,
                             attn_drop=self.attn_drop_rate, drop_path=self.dpr[i]) for i in range(self.depth)]
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.get_item = tf.keras.layers.Lambda(lambda x:x[:, 0])
        self.build((1, 224, 224, 3))
    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed

        class_pos_embed = np.array(self.pos_embed[:, 0])
        patch_pos_embed = np.array(self.pos_embed[:, 1:])
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size

        patch_pos_embed = np.array(patch_pos_embed)
        #shape_patch_pos_embed = patch_pos_embed.shape
        patch_pos_embed = tf.image.resize(patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim), (w0, h0), method='bicubic').numpy()
        assert int(w0) == patch_pos_embed.shape[1] and int(h0) == patch_pos_embed.shape[2]

        patch_pos_embed = patch_pos_embed.reshape(1, patch_pos_embed.shape[1]*patch_pos_embed.shape[2], dim)
        return np.concatenate((np.expand_dims(class_pos_embed, 0), patch_pos_embed), axis=1)

    
    def call(self, x):
        w, h, nc = x.shape[1], x.shape[2], x.shape[3]
        print('input shape:', x.shape)
        x = self.patch_embed(x)
        print('patch_embed:', x.shape)
        print('cls_token:', self.cls_token.shape)
        y = tf.tile(self.cls_token, (tf.shape(x)[0], 1, 1))
        x = self.concat([y, x])
        #print('pos_encoding before interpolation:', x.shape)
        y = self.interpolate_pos_encoding(x, w, h)
        y = tf.tile(y, (tf.shape(x)[0], 1, 1))
        x = self.layer_add([x, y])
        print('embedding:', x.shape)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
            print('block:', x.shape)
        x = self.layer_norm(x)
        x = self.get_item(x)
        print('out_dim:', x.shape)
        return x

    def get_last_selfattention(self, x):
        w, h, nc = x.shape[1], x.shape[2], x.shape[3]
        x = self.patch_embed(x)
        y = tf.tile(self.cls_token, (tf.shape(x)[0], 1, 1))
        x = self.concat([y, x])
        y = self.interpolate_pos_encoding(x, w, h)
        y = tf.tile(y, (tf.shape(x)[0], 1, 1))
        x = self.pos_drop(x)
        for blk in self.blocks[:-1]:
            x = blk(x)
        attn = self.blocks[-1](x, return_attention=True)
        return attn

def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(patch_size=patch_size, embed_dim=192, depth=2,
                num_heads=3, mlp_ratio=4., qkv_bias=True, **kwargs)
    return model

def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(patch_size=patch_size, embed_dim=384, depth=12,
                num_heads=6, mlp_ratio=4., qkv_bias=True,**kwargs)
    return model

def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(patch_size=patch_size, embed_dim=768, depth=12,
                num_heads=12, mlp_ratio=4., qkv_bias=True, **kwargs)
    return model
    
def DINOHead(in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3,
            hidden_dim=2048, bottleneck_dim=256, **kwargs):
    nlayers = max(nlayers, 1)
    inputs = tf.keras.layers.Input((in_dim, ))
    if nlayers==1:
        x = tf.keras.layers.Dense(bottleneck_dim)(inputs)
    else:
        x = tf.keras.layers.Dense(hidden_dim)(inputs)
        if use_bn:
            x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5)(x)
        x = tf.keras.layers.Lambda(lambda x : tf.keras.activations.gelu(x))(x)
        for _ in range(nlayers - 2):
            x = tf.keras.layers.Dense(hidden_dim)(x)
            if use_bn:
                x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5)(x)
            x = tf.keras.layers.Lambda(lambda x : tf.keras.activations.gelu(x))(x)
        x = tf.keras.layers.Dense(bottleneck_dim)(x)
    x = tf.keras.layers.Lambda(lambda x:tf.math.l2_normalize(x, axis=0), **{'name':'L2_Norm'})(x)
    dense = tf.keras.layers.Dense(out_dim, use_bias=False)
    last_layer = tfa.layers.WeightNormalization(dense)
    last_layer.build(x.shape)
    
    if norm_last_layer:
        last_layer.g = last_layer.add_weight(
                name="g",
                shape=(last_layer.layer_depth,),
                initializer="ones",
                dtype=last_layer.layer.kernel.dtype,
                trainable=False,
            )

    x = last_layer(x)

    return tf.keras.Model(inputs, x, name='dino_head', **kwargs)
