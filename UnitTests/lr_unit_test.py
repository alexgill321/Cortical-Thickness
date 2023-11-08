""" This module contains unit tests for the learning rate schedulers.

    The unit tests are designed to test the learning rate schedulers in modelUtils.lr_utils.py. The tests are designed
    to test the learning rate schedulers in isolation, and not in conjunction with other components of the models.
    Therefore, the tests conducted here involve verifying the desired functionality of the learning rate schedulers,
    and the multi optimizer learning rate scheduler.

    The tests are designed to be run with the following command:
        python -m unittest UnitTests.lr_unit_tests
"""

import unittest
from models.aae_models import AAEOptimizer
import os
from utils import generate_data
from modelUtils.lr_utils import MultiOptimizerLearningRateScheduler, CyclicLR, ExponentialDecayScheduler


class TestSchedulerModels(unittest.TestCase):
    """ Unit tests for the learning rate schedulers.

    This class contains unit tests for the learning rate schedulers in modelUtils.lr_utils.py. The tests are designed
    to test the learning rate schedulers in isolation. The main purpose of these tests is to verify the desired
    functionality of the learning rate schedulers, and the multi optimizer learning rate scheduler.
    """
    def setUp(self):
        cur = os.getcwd()
        filepath = os.path.join(cur, '../outputs/megasample_cleaned.csv')
        self.train_data, self.val_data, self.test_data = generate_data(filepath)

    def test_decay_to_expected_value_and_stay(self):
        initial_lr = 0.1
        decay_rate = 0.95
        decay_steps = 10
        scheduler = ExponentialDecayScheduler(initial_lr, decay_rate, decay_steps)

        # Run decay_steps iterations
        for i in range(decay_steps):
            scheduler.step()

        expected_lr = initial_lr * decay_rate

        # Test if the learning rate reaches the expected value after decay_steps iterations
        self.assertAlmostEqual(scheduler.get_lr(), expected_lr)

        # Run an additional 5 iterations
        for _ in range(decay_steps):
            scheduler.step()

        expected_lr = initial_lr * decay_rate ** 2
        # Test if the learning rate stays at the expected value after the additional iterations
        self.assertAlmostEqual(scheduler.get_lr(), expected_lr)

    def test_triangular_mode(self):
        base_lr = 0.001
        max_lr = 0.006
        step_size = 100

        clr = CyclicLR(base_lr=base_lr, step_size=step_size, max_lr=max_lr, mode="triangular")

        # Test the increasing phase
        for _ in range(step_size):
            lr = clr.step()
            self.assertGreaterEqual(lr, clr.base_lr)
            self.assertLessEqual(lr, clr.max_lr)

        # Test the peak
        lr = clr.step()
        self.assertEqual(lr, clr.max_lr)

        # Test the decreasing phase
        for _ in range(step_size - 1):
            lr = clr.step()
            self.assertGreaterEqual(lr, clr.base_lr)
            self.assertLessEqual(lr, clr.max_lr)

        # Test the return to the base learning rate
        lr = clr.step()
        self.assertEqual(lr, clr.base_lr)

    def test_triangular2_mode(self):
        base_lr = 0.001
        max_lr = 0.006
        step_size = 100

        clr = CyclicLR(base_lr=base_lr, step_size=step_size, max_lr=max_lr, mode="triangular2")
        clr.step()
        for cycle in range(1, 4):
            # Test the increasing phase
            for _ in range(step_size-1):
                lr = clr.step()
                self.assertGreaterEqual(lr, clr.base_lr)
                self.assertLessEqual(lr, clr.max_lr)

            # Test the peak
            lr = clr.step()
            expected_max = clr.base_lr + (clr.max_lr-clr.base_lr) / (2 ** (cycle - 1))
            self.assertAlmostEqual(lr, expected_max, places=6)

            # Test the decreasing phase
            for _ in range(step_size - 1):
                lr = clr.step()
                self.assertGreaterEqual(lr, clr.base_lr)
                self.assertLessEqual(lr, clr.max_lr)
            lr = clr.step()
            self.assertEqual(lr, clr.base_lr)

    def test_exp_range_mode(self):
        base_lr = 0.001
        max_lr = 0.006
        step_size = 100
        gamma = 0.999

        clr = CyclicLR(base_lr=base_lr, step_size=step_size, max_lr=max_lr, mode="exp_range")
        clr.step()
        for _ in range(3):  # 3 cycles
            # Test the increasing phase
            for _ in range(step_size-1):
                lr = clr.step()
                self.assertGreaterEqual(lr, clr.base_lr)
                self.assertLessEqual(lr, clr.max_lr)

            # Test the peak
            lr = clr.step()
            expected_max = clr.base_lr + (clr.max_lr - clr.base_lr) * (gamma ** (clr.iteration - 1))
            self.assertAlmostEqual(lr, expected_max, places=6)

            # Test the decreasing phase
            for _ in range(step_size - 1):
                lr = clr.step()
                self.assertGreaterEqual(lr, clr.base_lr)
                self.assertLessEqual(lr, clr.max_lr)

            # Test the return to the base learning rate
            lr = clr.step()
            self.assertEqual(lr, clr.base_lr)

    def test_multi_optimizer_lr_scheduler(self):
        base_lr = 0.001
        max_lr = 0.006
        step_size = 100
        gamma = 0.999

        opt = AAEOptimizer()
        autoencoder_scheduler = CyclicLR(base_lr=base_lr, step_size=step_size, max_lr=max_lr, mode="exp_range")
        generator_scheduler = CyclicLR(base_lr=base_lr, step_size=step_size, max_lr=max_lr, mode="triangular2")
        discriminator_scheduler = ExponentialDecayScheduler(initial_lr=base_lr, decay_rate=gamma,
                                                            decay_steps=step_size)

        optimizers = [opt.autoencoder_optimizer, opt.generator_optimizer, opt.discriminator_optimizer]
        schedulers = [autoencoder_scheduler, generator_scheduler, discriminator_scheduler]
        lr_and_opt = zip(schedulers, optimizers)
        scheduler = MultiOptimizerLearningRateScheduler(lr_and_opt)

        # Test the increasing phase
        for _ in range(step_size-1):
            scheduler.step()
            for optimizer, lr in lr_and_opt:
                self.assertEqual(optimizer.learning_rate, lr.get_lr())

        # Test the peak
        scheduler.step()
        for optimizer, lr in lr_and_opt:
            self.assertEqual(optimizer.learning_rate, lr.get_lr())

        # Test the decreasing phase
        for _ in range(step_size-1):
            scheduler.step()
            for optimizer, lr in lr_and_opt:
                self.assertEqual(optimizer.learning_rate, lr.get_lr())

        # Test the return to the base learning rate
        scheduler.step()
        for optimizer, lr in lr_and_opt:
            self.assertEqual(optimizer.learning_rate, lr.get_lr())


if __name__ == '__main__':
    unittest.main()
