package com.jujutsu.tsne;

import static com.jujutsu.utils.EjmlOps.addRowVector;
import static com.jujutsu.utils.EjmlOps.assignAllLessThan;
import static com.jujutsu.utils.EjmlOps.assignAtIndex;
import static com.jujutsu.utils.EjmlOps.biggerThan;
import static com.jujutsu.utils.EjmlOps.colMean;
import static com.jujutsu.utils.EjmlOps.maximize;
import static com.jujutsu.utils.EjmlOps.replaceNaN;
import static com.jujutsu.utils.EjmlOps.setData;
import static com.jujutsu.utils.EjmlOps.setDiag;
import static com.jujutsu.utils.EjmlOps.tile;
import static com.jujutsu.utils.MatrixOps.abs;
import static com.jujutsu.utils.MatrixOps.equal;
import static com.jujutsu.utils.MatrixOps.fillMatrix;
import static com.jujutsu.utils.MatrixOps.negate;
import static com.jujutsu.utils.MatrixOps.range;
import static com.jujutsu.utils.MatrixOps.rnorm;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;

import com.jujutsu.utils.MatrixOps;
/**
 *
 * Author: Leif Jonsson (leif.jonsson@gmail.com)
 * 
 * This is a Java implementation of van der Maaten and Hintons t-sne 
 * dimensionality reduction technique that is particularly well suited 
 * for the visualization of high-dimensional datasets
 *
 */
public class MemOptimizedTSne extends FastTSne {
	
	@Override
	public double [][] tsne(TSneConfiguration config) {
		double[][] X      = config.getXin();
		int no_dims       = config.getOutputDims();
		int initial_dims  = config.getInitialDims(); 
		double perplexity = config.getPerplexity();
		int max_iter      = config.getMaxIter();
		boolean use_pca   = config.usePca();
		

		String IMPLEMENTATION_NAME = this.getClass().getSimpleName();
		System.out.println("X:Shape is = " + X.length + " x " + X[0].length);
		System.out.println("Running " + IMPLEMENTATION_NAME + ".");
		long end = System.currentTimeMillis();
		long start = System.currentTimeMillis();
		// Initialize variables
		if(use_pca && X[0].length > initial_dims && initial_dims > 0) {
			PrincipalComponentAnalysis pca = new PrincipalComponentAnalysis();
			X = pca.pca(X, initial_dims);
			System.out.println("X:Shape after PCA is = " + X.length + " x " + X[0].length);
			System.out.println(MatrixOps.doubleArrayToPrintString(X,10,10));
		}
		int n = X.length;
		double momentum = .5;
		double initial_momentum = 0.5;
		double final_momentum   = 0.8;
		int eta                 = 500;
		double min_gain         = 0.01;
		DMatrixRMaj Y        = new DMatrixRMaj(rnorm(n,no_dims));
		DMatrixRMaj Ysqlmul  = new DMatrixRMaj(Y.numRows,Y.numRows); // Ysqlmul = n x n
		DMatrixRMaj dY       = new DMatrixRMaj(fillMatrix(n,no_dims,0.0));
		DMatrixRMaj iY       = new DMatrixRMaj(fillMatrix(n,no_dims,0.0));
		DMatrixRMaj gains    = new DMatrixRMaj(fillMatrix(n,no_dims,1.0));
		DMatrixRMaj btNeg    = new DMatrixRMaj(n,no_dims);
		DMatrixRMaj bt       = new DMatrixRMaj(n,no_dims);
		
		// Compute P-values
		DMatrixRMaj P        = new DMatrixRMaj(x2p(X, 1e-5, perplexity).P); // P = n x n
		DMatrixRMaj Psized   = new DMatrixRMaj(P.numRows,P.numCols);        // L = n x n
		DMatrixRMaj diag     = new DMatrixRMaj(fillMatrix(Psized.numRows,Psized.numCols,0.0));
		
		CommonOps_DDRM.transpose(P,Psized);
		CommonOps_DDRM.addEquals(P,Psized);
		CommonOps_DDRM.divide(P, CommonOps_DDRM.elementSum(P));
		replaceNaN(P,Double.MIN_VALUE);
		CommonOps_DDRM.scale(4.0,P);					// early exaggeration
		maximize(P, 1e-12);
		
		System.out.println("Using perplexity: " + perplexity);
		System.out.println("Y:Shape is = " + Y.getNumRows() + " x " + Y.getNumCols());

		DMatrixRMaj sqed  = new DMatrixRMaj(Y.numRows,Y.numCols);  // sqed = n x n
		DMatrixRMaj sum_Y = new DMatrixRMaj(1,Y.numRows);
		DMatrixRMaj Q     = new DMatrixRMaj(P.numRows,P.numCols);  // Q = n x n
		
		for (int iter = 0; iter < max_iter; iter++) {
			// Compute pairwise affinities
			CommonOps_DDRM.elementPower(Y, 2, sqed);
			CommonOps_DDRM.sumRows(sqed, sum_Y);
			CommonOps_DDRM.multAddTransB(-2.0, Y, Y, Ysqlmul);
			addRowVector(Ysqlmul, sum_Y);
			CommonOps_DDRM.transpose(Ysqlmul);
			addRowVector(Ysqlmul, sum_Y);
			
			CommonOps_DDRM.add(Ysqlmul, 1.0);
			CommonOps_DDRM.divide(1.0,Ysqlmul);
			assignAtIndex(Ysqlmul, range(n), range(n), 0);
			CommonOps_DDRM.divide(Ysqlmul , CommonOps_DDRM.elementSum(Ysqlmul), Q);

			maximize(Q, 1e-12);
			
			// Compute gradient
			CommonOps_DDRM.subtract(P, Q, Psized);
			CommonOps_DDRM.elementMult(Psized, Ysqlmul);
			DMatrixRMaj rowsum = CommonOps_DDRM.sumRows(Psized,null); // rowsum = nx1
			double [] rsum  = new double[rowsum.numRows];
			for (int i = 0; i < rsum.length; i++) {
				rsum[i] = rowsum.get(i,0);
			}
			setDiag(diag,rsum);
			CommonOps_DDRM.subtract(diag, Psized, Psized);
			CommonOps_DDRM.mult(Psized, Y, dY);
			CommonOps_DDRM.scale(4.0, dY);
			
			// Perform the update
			if (iter < 20)
				momentum = initial_momentum;
			else
				momentum = final_momentum;
			
			boolean [][] boolMtrx = equal(biggerThan(dY,0.0),biggerThan(iY,0.0));
			
			
			setData(btNeg, abs(negate(boolMtrx)));
			setData(bt, abs(boolMtrx));
			
			DMatrixRMaj gainsSmall = new DMatrixRMaj(gains);
			DMatrixRMaj gainsBig   = new DMatrixRMaj(gains);
			CommonOps_DDRM.add(gainsSmall,0.2);
			CommonOps_DDRM.scale(0.8,gainsBig);
			
			CommonOps_DDRM.elementMult(gainsSmall, btNeg);
			CommonOps_DDRM.elementMult(gainsBig, bt);
			CommonOps_DDRM.add(gainsSmall,gainsBig,gains);

			assignAllLessThan(gains, min_gain, min_gain);
			
			CommonOps_DDRM.scale(momentum,iY);
			DMatrixRMaj gainsdY = new DMatrixRMaj(gains.numRows,dY.numCols);
			CommonOps_DDRM.elementMult(gains , dY, gainsdY);
			CommonOps_DDRM.scale(eta,gainsdY);
			CommonOps_DDRM.subtractEquals(iY , gainsdY);
			CommonOps_DDRM.addEquals(Y , iY);
			DMatrixRMaj colMeanY = colMean(Y, 0);
			DMatrixRMaj meanTile = tile(colMeanY, n, 1);
			CommonOps_DDRM.subtractEquals(Y , meanTile);

			// Compute current value of the cost function
			if (iter % 50 == 0)   {
				DMatrixRMaj Pdiv = new DMatrixRMaj(P);
				CommonOps_DDRM.elementDiv(Pdiv , Q);
				CommonOps_DDRM.elementLog(Pdiv,Psized);
				replaceNaN(Psized,Double.MIN_VALUE);
				CommonOps_DDRM.elementMult(Psized,P);
				replaceNaN(Psized,Double.MIN_VALUE);
				double C = CommonOps_DDRM.elementSum(Psized);
				end = System.currentTimeMillis();
				System.out.printf("Iteration %d: error is %f (50 iterations in %4.2f seconds)\n", iter, C, (end - start) / 1000.0);
				if(C < 0) {
					System.err.println("Warning: Error is negative, this is usually a very bad sign!");
				}
				start = System.currentTimeMillis();
			} else if(iter % 10 == 0) {
				end = System.currentTimeMillis();
				System.out.printf("Iteration %d: (10 iterations in %4.2f seconds)\n", iter, (end - start) / 1000.0);
				start = System.currentTimeMillis();
			}

			// Stop lying about P-values
			if (iter == 100)
				CommonOps_DDRM.divide(P , 4);
		}

		// Return solution
		return MatrixOps.extractDoubleArray(Y);
	}
}
