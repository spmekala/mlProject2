package shared;

import java.util.List;

/**
 * A fixed iteration trainer
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class FixedIterationTrainer implements Trainer {
    
    /**
     * The inner trainer
     */
    private Trainer trainer;
    
    /**
     * The number of iterations to train
     */
    private int iterations;
    
    /**
     * Make a new fixed iterations trainer
     * @param t the trainer
     * @param iter the number of iterations
     */
    public FixedIterationTrainer(Trainer t, int iter) {
        trainer = t;
        iterations = iter;
    }

    /**
     * @see shared.Trainer#train()
     */
    public double train() {
        double sum = 0;
        for (int i = 0; i < iterations; i++) {
            sum += trainer.train();
        }
        return sum / iterations;
    }

    public double train(List<Double> lines, List<Long> times) {
        double sum = 0;
        long start = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            double fitness = trainer.train();
            lines.add(fitness);
            times.add(System.nanoTime() - start);
            sum += fitness;
        }
        return sum / iterations;
    }

}
