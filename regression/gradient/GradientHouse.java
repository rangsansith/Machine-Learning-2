package hw2.regression.gradient;

import java.io.BufferedReader;
import java.io.FileReader;

import Jama.Matrix;

public class GradientHouse {

	private static double hofw(double w[],double x[],double y){
		double ans=0.0;
		for (int i = 0; i < w.length; i++) {
			ans+=w[i]*x[i];
			//System.out.println(ans+"    "+w[i]+"    "+x[i]);
		}
		ans-=y;
		return ans;
	}
	public static void main(String[] args) {
		double[][] features = new double[433][14];
		double [][] labels = new double[433][1];
		double[] mean= new double[13];
		double[] std= new double[13];
		for(int i=0;i<13;i++)
		{
			mean[i]=0.0;
		}
		FileReader featureread;
		try {
			featureread = new FileReader("data/housing_train.txt");
			BufferedReader featurereadbr = new BufferedReader(featureread);
			String sCurrentLine;
			String[] feats;
			int ind=0;
			while ((sCurrentLine = featurereadbr.readLine()) != null) {
				sCurrentLine = sCurrentLine.trim().replaceAll(" +", " ");
				feats=sCurrentLine.split(" ");
				for(int i=0;i<feats.length-1;i++)
				{
					features[ind][i]=Double.parseDouble(feats[i]);
					mean[i]+=features[ind][i];
				}
				features[ind][feats.length-1]=1.0;
				labels[ind][0]=Double.parseDouble(feats[feats.length-1]);
				ind++;
			}
			featurereadbr.close();
			featureread.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		for(int i=0;i<13;i++)
		{
			mean[i]/=433.0;
		}
		for (int i = 0; i < mean.length; i++) {
			std[i]=0.0;
			for (int j = 0; j < features.length; j++) {
				std[i]+=(features[j][i]-mean[i])*(features[j][i]-mean[i]);
			}
			std[i]=Math.sqrt(std[i]);
		}
		for (int i = 0; i < features.length; i++) {
			for (int j = 0; j < mean.length; j++) {
				features[i][j]-=mean[j];
				features[i][j]/=std[j];
			}
		}
		double [][] featurestest= new double[74][14];
		double [][] labelstest= new double[74][1];
		try {
			featureread = new FileReader("data/housing_test.txt");
			BufferedReader featurereadbr = new BufferedReader(featureread);
			String sCurrentLine;
			String[] feats;
			int ind=0;
			while ((sCurrentLine = featurereadbr.readLine()) != null) {
				sCurrentLine = sCurrentLine.trim().replaceAll(" +", " ");
				feats=sCurrentLine.split(" ");
				for(int i=0;i<feats.length-1;i++)
				{
					featurestest[ind][i]=(Double.parseDouble(feats[i])-mean[i])/std[i];
				}
				featurestest[ind][feats.length-1]=1.0;
				labelstest[ind][0]=Double.parseDouble(feats[feats.length-1]);
				ind++;
			}
			featurereadbr.close();
			featureread.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		Matrix x = new Matrix(features);
		Matrix y = new Matrix(labels);
		
		double w[]=new double[14];
		double lambda=0.0005;
		for (int i = 0; i < w.length-1; i++) {
			w[i]=3.2;
		}
		w[13]=20.2;
		double prev=0.0,pres=10000.0,thresh=0.0000007,ctr=0.0;
		int iter=0;
		while(Math.abs(prev-pres)>thresh) {
			if(iter>0&&pres>prev&&ctr<26)
				break;
			System.out.println(iter++);
			prev=pres;
			for (int i = 0; i < features.length; i++) {
				for (int j = 0; j < w.length; j++) {
					//System.out.println("before "+i+"  "+j+"  "+w[j]);
					w[j]=w[j]-lambda*hofw(w,features[i],labels[i][0])*features[i][j];
					//System.out.println("after "+i+"  "+j+"  "+w[j]);

				}
			}
			Matrix wma=new Matrix(w, w.length);
			Matrix xtest=new Matrix(featurestest);
			Matrix ytest = new Matrix(labelstest);
			Matrix mul3 = (xtest.times(wma));
			//System.out.println(mul3.getColumnDimension()+"  "+mul3.getRowDimension());
			Matrix error= mul3.minus(ytest);
			Matrix fete=(error.transpose().times(error)).times(1);
			//fete.print(1, 6);
			System.out.println("Testing error:"+fete.get(0, 0)/74.0);
			pres=fete.get(0, 0)/74.0;
			Matrix mul4 = x.times(wma);
			Matrix errortrain= mul4.minus(y);
			Matrix fet=(errortrain.transpose().times(errortrain)).times(1);
			//fet.print(1, 6);
			System.out.println("Training error:"+fet.get(0, 0)/433.0);
			ctr=fet.get(0, 0)/433.0;
		}
		System.out.println(prev+"  "+pres);
		for (int i = 0; i < w.length; i++) {
			//System.out.println(w[i]);
		}
	}

}
