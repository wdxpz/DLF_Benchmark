from dc_gan import parser, train

args = parser.parse_args()

for i in range(3):
    train(args)
