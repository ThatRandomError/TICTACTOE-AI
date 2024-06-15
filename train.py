import pyn

dataset = [
    [
        [0,0,0, ### INPUT
        0,0,0,
        0,0,0],

        [0,0,0, ### EXPECTED OUTPUT
        0,1,0,
        0,0,0]
    ],
    [
        [0,0,0, ### INPUT
        1,0,1,
        0,0,0],

        [0,0,0, ### EXPECTED OUTPUT
        0,1,0,
        0,0,0]
    ],
]

network = pyn.NN([9,12,9]) #### NN STRUCT ### INPUT LAYER: 9 ### HIDDEN LAYER: 12 ### OUTPUT LAYER: 9

network.randomize_values()

network.load_dataset(dataset)

network.train(5000,0.1,printdata=True)

network.save("ttt.pyn")

network.accuracy()