package hw3.generative.em;

import java.io.BufferedReader;
import java.io.FileReader;
import Jama.Matrix;

public class EM {
	static int datadime=2;
	static int datalen=6000;
	static int nofgaussian=3;
	static double[][] data;
	static double[][] z;
	static double[] pi;
	static double pxzt=0.0;
	static double[] ztot;
	static double[] pxz;
	static Matrix[] mu;
	static Matrix[] covar;
	static Matrix[] icovar;
	static double[] det;
	static double preverror;
	static double errorthreshold=0.000000001;
	private static void initdata(){
		data= new double[datalen][datadime];
		z=new double[datalen][nofgaussian];
		pi=new double[nofgaussian];
		ztot=new double[nofgaussian];
		pxz=new double[nofgaussian];
		mu = new Matrix[nofgaussian];
		covar=new Matrix[nofgaussian];
		icovar=new Matrix[nofgaussian];
		det= new double[nofgaussian];
			
	}
	private static void readData(int gaussno){
		nofgaussian=gaussno;
		if(gaussno==3)
			  datalen=10000;
		initdata();
		try {
			FileReader featureread = new FileReader("data/"+gaussno+"gaussian.txt");
			BufferedReader featurereadbr = new BufferedReader(featureread);
			String sCurrentLine;
			String[] feats;
			int ind=0;
			while ((sCurrentLine = featurereadbr.readLine()) != null) {
				feats=sCurrentLine.split(" ");
				for(int i=0;i<feats.length;i++)
				{
					data[ind][i]=Double.parseDouble(feats[i]);

				}
				ind++;
			}
			featurereadbr.close();
			featureread.close();
		}catch(Exception e){
			e.printStackTrace();
		}
	}
	private static void dispMAtrix(Matrix a){
		for (int i = 0; i < a.getRowDimension(); i++) {
			for (int j = 0; j < a.getColumnDimension(); j++) {
				System.out.print(a.get(i, j)+"  ");
			}
			System.out.println();
		}
	}
	private static double calculateGM(Matrix x,Matrix mu,Matrix icovar,double det){
		Matrix xmmu=x.minus(mu);
		Matrix xmmut=xmmu.transpose();
		Matrix total=(xmmu.times(icovar)).times(xmmut);
		return Math.exp(-0.5*total.det())/Math.sqrt(Math.pow(2.0*Math.PI,datadime)*det);
	}
	private static void dispResult()
	{
		for (int i = 0; i < nofgaussian; i++) {
			System.out.println("Gaussian"+(i+1));
			System.out.println("Mean");
			dispMAtrix(mu[i]);
			System.out.println("CoVar");
			dispMAtrix(covar[i]);
			System.out.println("Pi "+pi[i]);
			System.out.println("N: "+ztot[i]);
			System.out.println();
		}
	}
	private static boolean checkerror()
	{
		double logerror=0.0;
		for (int i = 0; i < datalen; i++) {
			double sum=0.0;
			for (int j = 0; j < nofgaussian; j++) {
				sum+=pi[j]*calculateGM(new Matrix(data[i],1), mu[j], icovar[j], det[j]);
			}
			logerror+=Math.log(sum);
		}
		if(Math.abs(logerror-preverror)<errorthreshold)
			return false;
		else
			{
			preverror=logerror;
			return true;
			}
	}
	private static void init(){
		//initialisation
				for (int k = 0; k < nofgaussian; k++) {
					for (int i = 0; i < datalen; i++) {
						z[i][k]=0;	
					}
					pi[k]=0;
				}
				// z equal
				for (int j = 0; j < nofgaussian; j++) {
					for (int i = (j)*(datalen/nofgaussian); i < (j+1)*(datalen/nofgaussian); i++) {
						z[i][nofgaussian-j-1]=1;
					}
				}
				// pi
				for (int j = 0; j < nofgaussian; j++) {
					for (int i = 0; i < datalen; i++) {
						pi[j]+=z[i][j];
					}	
					pi[j]/=datalen;
				}
				//mu calculation
				for (int i = 0; i < nofgaussian; i++) {
					ztot[i]=0.0;
					mu[i]=new Matrix(1,datadime);
					for (int j = 0; j < datalen; j++) {
						Matrix x=new Matrix(data[j], 1);
						ztot[i]+=z[j][i];
						mu[i].plusEquals(x.times(z[j][i]));
					}
					mu[i].timesEquals(1.0/ztot[i]);
				}
				// covar calculation
				for (int i = 0; i < nofgaussian; i++) {
					covar[i]=new Matrix(datadime,datadime);
					for (int j = 0; j < datalen; j++) {
						Matrix x=new Matrix(data[j], 1);
						Matrix xmmu=x.minus(mu[i]);
						Matrix xmmut=x.transpose();
						covar[i].plusEquals(xmmut.times(xmmu).timesEquals(z[j][i]));
					}
					covar[i].timesEquals(1.0/ztot[i]);
					icovar[i]=covar[i].inverse();
					det[i]=covar[i].det();
				}
				
	}
    private static void E()
    {
    	// E step
    				for (int i = 0; i < nofgaussian; i++) {
    					ztot[i]=0;
    				}
    				for (int i = 0; i < datalen; i++) {
    					pxzt=0.0;
    					Matrix x=new Matrix(data[i], 1);
    					for (int j = 0; j < nofgaussian; j++) {
    						pxz[j]=calculateGM(x, mu[j], icovar[j], det[j]);
    						pxzt+=pxz[j]*pi[j];
    						//System.out.println(i+" b "+z[i][j]+"  "+pxz[j]+"  "+pxzt);
    					}
    					for (int j = 0; j < nofgaussian; j++) {
    						z[i][j]=(pxz[j]*pi[j])/(pxzt);
    						ztot[j]+=z[i][j];
    						//System.out.println(i+" a "+z[i][j]+"  "+pxz[j]+"  "+pxzt);
    					}
    				}
    }
	private static void M(){
		// M step
					for (int i = 0; i < nofgaussian; i++) {
						mu[i]=new Matrix(1,datadime);
						for (int j = 0; j < datalen; j++) {
							Matrix x=new Matrix(data[j], 1);
							mu[i].plusEquals(x.times(z[j][i]));
						}
						mu[i].timesEquals(1.0/ztot[i]);
					}
					for (int i = 0; i < nofgaussian; i++) {
						covar[i]=new Matrix(datadime,datadime);
						for (int j = 0; j < datalen; j++) {
							Matrix x=new Matrix(data[j], 1);
							Matrix xmmu=x.minus(mu[i]);
							Matrix xmmut=x.transpose();
							covar[i].plusEquals(xmmut.times(xmmu).timesEquals(z[j][i]));
						}
						covar[i].timesEquals(1.0/ztot[i]);
						pi[i]=ztot[i]/((double) datalen);
						icovar[i]=covar[i].inverse();
						det[i]=covar[i].det();
					}
	}
    public static void main(String[] args) {
		readData(3);
		init();
		int iter=0,maxiter=500;
		dispResult();
		while(checkerror()&&iter<maxiter){
			E();
			M();
			iter++;
		}
		System.out.println("Stopped :"+iter);
		//display results;
		dispResult();
	}

}
