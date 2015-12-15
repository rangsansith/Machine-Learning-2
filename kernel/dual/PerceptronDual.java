package hw7.kernel.dual;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

public class PerceptronDual {

	static int len=1000;
	static int featlen=4;
	static double[][] features = new double[len][featlen];
	static double[] labels = new double[len];
	static double[] alpha=new double[len];
	static double [][] kernelmat= new double[len][len];
	static int kerneltype=2;
	static double sigmag=1.0/featlen;	
	static double tanc=0.1;
	static double tanb=0.0;
	static double b=0;
	private static double dotproduct(double[] i,double[] j){
		double ans=0;
		for(int k=0;k<featlen;k++)
		{
			ans+=i[k]*j[k];
		}
		return ans;
	}
	private static double euclidist(double[] i, double[] j) {
		double sum=0;
		for (int k=0;k<featlen;k++) {
			sum+=(i[k]-j[k])*(i[k]-j[k]);
		}
		return sum;
	}
	private static double kernel(double[] i, double[] j){
		if(kerneltype==1)
		{
			return dotproduct(i, j);
		}
		else if(kerneltype==2)
		{
			return Math.exp(-(euclidist(i, j)*(sigmag*sigmag)));
		}
		else if(kerneltype==3)
		{
			return Math.tanh(sigmag*dotproduct(i, j)+tanb);
		}
		else
		{
			return Math.pow(sigmag*dotproduct(i, j)+tanb, tanc);
		}
	}
	private static double mpredict(int xno){
		double ans=b;
		double []kern=kernelmat[xno];
		for (int i = 0; i < len; i++) {
			ans+=alpha[i]*kern[i]*labels[i];
		}
		return ans;
	}
	
    private static double mpredict(double [] xno){
		double ans=0.0;
		for (int i = 0; i < len; i++) {
			ans+=alpha[i]*kernel(xno,features[i]);
		}
		return ans;
	}
	private static void readata(){
		try {
			FileReader featureread = new FileReader("data/twoSpirals.txt");
			BufferedReader featurereadbr = new BufferedReader(featureread);
			String sCurrentLine;
			String[] feats;
			int ind=0;
			while ((sCurrentLine = featurereadbr.readLine()) != null) {
				feats=sCurrentLine.split("	");
				labels[ind]=Double.parseDouble(feats[feats.length-1]);
				for(int i=0;i<feats.length-1;i++)
				{
					features[ind][i]=Double.parseDouble(feats[i]);
				}
				//features[ind][featlen-1]=1;
				ind++;
			}
			featurereadbr.close();
			featureread.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		
	}
	private static void initalphas(){
		for (int i = 0; i < len; i++) {
			alpha[i]=0.0;
		}
	}
	private static void setkernel(){
		for(int i=0;i<len;i++)
		{
			for(int j=0;j<len;j++)
			{
				kernelmat[i][j]=kernel(features[i],features[j]);
			}
		}
	}
	
    public static void main(String[] args) {
		readata();
		initalphas();
		setkernel();
		int mistakes=len;
		while(mistakes>0)
		{
			mistakes=0;
			for (int i = 0; i < len; i++) {
			 double t =mpredict(i);
			 if(t*labels[i]<=0)
			 {
				 mistakes++;
				 alpha[i]++;
				 b+=labels[i];
			 }
			}
			System.out.println(mistakes);
		}
	}

}
