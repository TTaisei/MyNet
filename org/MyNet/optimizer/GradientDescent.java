package org.MyNet.optimizer;

import MyNet.network.*;
import MyNet.costFunction.*;
import MyNet.matrix.*;

/**
 * Class for Gradient Descent.
 */
public class GradientDescent extends Optimizer{
    /**
     * Constructor for this class.
     */
    public GradientDescent(){
        ;
    }

    /**
     * Constructor for this class.
     * @param net Network to which optimization is applied.
     * @param f Cost function in this net.
     */
    public GradientDescent(Network net, CostFunction f){
        this.net = net;
        this.cFunc = f;
    }

    /**
     * Constructor for this class.
     * @param net Network to which optimization is applied.
     * @param f Cost function in this net.
     * @param eta Learning rate.
     */
    public GradientDescent(Network net, CostFunction f, double eta){
        this.net = net;
        this.cFunc = f;
        this.eta = eta;
    }

    /**
     * Run learning.
     * @param x Input layer.
     * @param t Answer.
     * @param nEpoch Number of epoch.
     * @return Output of this network.
     */
    public Matrix fit(Matrix x, Matrix t, int nEpoch){
        Matrix y = this.forward(x);

        for (int i = 0; i < nEpoch; i++){
            System.out.printf("Epoch %d/%d\n", i+1, nEpoch);
            this.back(x, y, t);
            y = this.forward(x);
            System.out.printf("loss: %.4f\n", this.cFunc.calcurate(y, t).matrix[0][0]);
        }

        return y;
    }

    /**
     * Run learning.
     * @param x Input layer.
     * @param t Answer.
     * @param nEpoch Number of epoch.
     * @param valX Input layer for validation.
     * @param valT Answer for validation.
     * @return Output of this network.
     */
    public Matrix fit(Matrix x, Matrix t, int nEpoch, Matrix valX, Matrix valT){
        Matrix y = this.forward(x);
        Matrix valY;

        for (int i = 0; i < nEpoch; i++){
            System.out.printf("Epoch %d/%d\n", i+1, nEpoch);
            this.back(x, y, t);
            valY = this.forward(valX);
            y = this.forward(x);
            System.out.printf(
                "loss: %.4f - valLoss: %.4f\n",
                this.cFunc.calcurate(y, t).matrix[0][0],
                this.cFunc.calcurate(valY, valT).matrix[0][0]
            );
        }

        return y;
    }
}