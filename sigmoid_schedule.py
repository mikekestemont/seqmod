
import math


def generic_sigmoid(a=1, b=1, c=1):
    return lambda x: a/(1 + b * math.exp(-x * c))


def get_b(inflection, c):
    """
    b = exp(inflection * c)
    """
    return math.exp(inflection * c)


def get_c(inflection, b):
    """
    c = log(b) / inflection
    """
    return math.log(b) / inflection


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env', default='sigmoid_schedule')
    parser.add_argument('--inflection', default=5000, type=int)
    args = parser.parse_args()

    from visdom import Visdom
    viz = Visdom(env=args.env)

    import numpy as np
    Y = np.linspace(1, args.inflection * 2, 1000)

    for b in (10, 100, 1000):
        c = get_c(args.inflection, b)
        title = 'c=%.g;b=%d;inflection=%d' % (c, b, args.inflection)
        viz.line(np.array([generic_sigmoid(b=b, c=c)(i) for i in Y]), Y,
                 opts={'title': title})
    for c in (0.0001, 0.001, 0.005):
        b = get_b(args.inflection, c)
        title = 'c=%.g;b=%d;inflection=%d' % (c, b, args.inflection)
        viz.line(np.array([generic_sigmoid(b=b, c=c)(i) for i in Y]), Y,
                 opts={'title': title})
