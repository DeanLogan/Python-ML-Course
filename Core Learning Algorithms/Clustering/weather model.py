import tensorflow_probability as tfp
import tensorflow as tf

tfd = tfp.distributions # making a shortcut for later on 
initial_distribution = tfd.Categorical(probs=[0.8, 0.2]) # 80% of being cold therefore 20% of being a warm day
transition_distribution = tfd.Categorical(probs=[[0.7,0.3], [0.2,0.8]]) # cold day has 30% of being followed be a hot day, then a hot day has a 20% of being followed by a cold day
observation_distribution = tfd.Normal(loc=[0.0, 15.0], scale=[5.0, 10.0]) # on each day the temp is normally distributed with mean and standard deviation 0 and 5 on a cold day and mean and standard deviation 15 and 10 on a hot day

# the loc argument represents the mean and the scale is the standard devitation

model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7
)

mean = model.mean()

with tf.compat.v1.Session() as sess:
    print(mean.numpy())