
//<![CDATA[

// SELF-NOTE: dunno how to get a Conv layer up in this.. couldn't quickly
//            figure out how to format in/output tensor shape -- probably
//            why it wasn't working. Apparently someone who did well last
//            year basically brute-forced this? https://github.com/jordanott/Deep-Traffic/blob/master/deep_traffic.js
//            I guess I'll see what a bigger FullNet can do since I'm
//            already close to the passing 65mph. May change if I see
//            working syntax for Conv->Pool->Full somewhere for ConvNet.js.
//              -- WNixalo

// a few things don't have var in front of them - they update already existing variables the game needs
lanesSide = 3;
patchesAhead = 17;
patchesBehind = 1;
trainIterations = 35000;

// the number of other autonomous vehicles controlled by your network
otherAgents = 0; // max of 10

var num_inputs = (lanesSide * 2 + 1) * (patchesAhead + patchesBehind);
var num_actions = 5;
var temporal_window = 3;
var network_size = num_inputs * temporal_window + num_actions * temporal_window + num_inputs;

var layer_defs = [];
    layer_defs.push({
    type: 'input',
    out_sx: 1,
    out_sy: 1,
    out_depth: network_size
});
// layer_defs.push({
//     type: 'conv',
//     filters: 5,
//     stride:1,
//     activation='relu'
// });
// layers_defs.push({
//     type:'pool',
//     sx:2,
//     stride:2})
layer_defs.push({
    type: 'fc',
    num_neurons: parseInt(network_size / num_actions / 1),
    activation: 'relu'
});
// Local Contrast Normalization https://cs.stanford.edu/people/karpathy/convnetjs/docs.html
layer_defs.push(
    {type:'lrn',
    k:1,
    n:3,
    alpha:0.1,
    beta:0.75});
layer_defs.push({
    type: 'fc',
    num_neurons: parseInt(network_size / num_actions / 1),
    activation: 'relu'
});
layer_defs.push({
    type: 'regression',
    num_neurons: num_actions
});

var tdtrainer_options = {
    learning_rate: 0.001,
    momentum: 0.0,
    batch_size: 128,
    l2_decay: 0.01
};

var opt = {};
opt.temporal_window = temporal_window;
opt.experience_size = 3000;
opt.start_learn_threshold = 500;
opt.gamma = 0.7;
opt.learning_steps_total = 10000;
opt.learning_steps_burnin = 1000;
opt.epsilon_min = 0.0;
opt.epsilon_test_time = 0.0;
opt.layer_defs = layer_defs;
opt.tdtrainer_options = tdtrainer_options;

brain = new deepqlearn.Brain(num_inputs, num_actions, opt);

learn = function (state, lastReward) {
brain.backward(lastReward);
var action = brain.forward(state);

draw_net();
draw_stats();

return action;
}

//]]>
