package hw2.nnet.perceptron;

import java.io.IOException;

public class NNNew {

	private static double sigmoid(double e)
	{
		return 1.0/(1.0+Math.exp(-e));
	}
	private static double sigmoidash(double e)
	{
		return sigmoid(e)*(1-sigmoid(e));
	}
	public static void main(String[] args) {
		int d=8,input=8,output=8,hidden=3;
		double x[][] = new double[d][input];
		double t[][]= new double[d][output];
		double wij[][] = new double[input][hidden];
		double bij[] = new double[hidden];
		double bjk[] = new double[output];
		double netj[] = new double[hidden];
		double netk[] = new double [output];
		double yj[] = new double[hidden];
		double zk[] = new double[output];

		for (int i = 0; i < d; i++) {
			for (int j = 0; j < input; j++) {
				if(i==j)
				{
					x[i][j]=1;
					t[i][j]=1;
				}
				else
				{
					x[i][j]=0;
					t[i][j]=0;
				}
			}
		}
		for (int i = 0; i < input; i++) {
			for (int j = 0; j < hidden; j++) {
				wij[i][j]=0.1;
			}
		}
		for (int i = 0; i < hidden; i++) {
			bij[i]=0.1;
		}
		for (int i = 0; i < output; i++) {
			bjk[i]=0.1;
		}
		double lambda=0.95,tjw=0.0,jw=0.0;
		while(true)
		{
			tjw=0.0;
			for (int n = 0; n < d; n++) {
				jw=0.0;
				// Input layer to Hidden layer
				for (int j = 0; j < hidden; j++) {
					netj[j]=bij[j];
					for (int i = 0; i < input; i++) {
						netj[j]+=x[n][i]*wij[i][j];
					}
					yj[j]=sigmoid(netj[j]);
				}
				// Hidden layer to Output layer
				for (int k = 0; k < output; k++) {
					netk[k]=bjk[k];
					for (int j = 0; j < hidden; j++) {
						netk[k]+=wij[k][j]*yj[j];
					}
					zk[k]=sigmoid(netk[k]);
					jw+=Math.pow((t[n][k]-zk[k]), 2);	
				}
				jw*=0.5;
				tjw+=jw;
				System.out.println(n);
				System.out.print("Input:");
				for(int i=0;i<input;i++)
				{
					System.out.print(" "+x[n][i]);
				}
				System.out.println();
				System.out.print("Output:");
				for(int i=0;i<output;i++)
				{
					System.out.print(" "+Math.floor(zk[i]+0.03));
				}
				System.out.println();
				System.out.print("Actual:");
				for(int i=0;i<output;i++)
				{
					System.out.print(" "+t[n][i]);
				}
				System.out.println();
				System.out.println("Step error: "+jw);
				if(jw>0)
				{
					// Backpropagation Output to Hidden
					for (int k = 0; k < output; k++) {
						for (int j = 0; j < hidden; j++) {
							wij[k][j]+=lambda*(t[n][k]-zk[k])*yj[j]*sigmoidash(netk[k]);
						}
						bjk[k]+=lambda*(t[n][k]-zk[k])*sigmoidash(netk[k]);
					}
					// Backpropagation Hidden to Input
					for (int j = 0; j < hidden; j++) {
						double temperror=0.0;
						for (int k = 0; k < output; k++) {
							temperror+=(t[n][k]-zk[k])*sigmoidash(netk[k])*wij[k][j];
						}
						for (int i = 0; i < input; i++) {
							wij[i][j]+=lambda*sigmoidash(netj[j])*x[n][i]*temperror;
						}
						bij[j]+=lambda*sigmoidash(netj[j])*temperror;
					}
				}
			}
			System.out.println(tjw);
			if(tjw<0.001)
				break;
			/*try {
				System.in.read();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}*/
		}
		System.out.println();
		for (int i = 0; i < input; i++) {
			for (int j = 0; j < hidden; j++) {
				System.out.print(wij[i][j]+"  ");
			}
			System.out.println();
		}
		System.out.println();
	}
}
