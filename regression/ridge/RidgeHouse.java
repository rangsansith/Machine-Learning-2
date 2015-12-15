package hw2.regression.ridge;

import java.io.BufferedReader;
import java.io.FileReader;

import Jama.Matrix;

public class RidgeHouse {

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
		double gte=100.0,gtr=100.0,cte=0.0,ctr=0.0,lambda=0.00001;
		while(true)
		{
			Matrix lam=Matrix.identity(14, 14).times(lambda);
			Matrix x = new Matrix(features);
			Matrix y = new Matrix(labels);
			Matrix xt = x.transpose();
			Matrix mul2=xt.times(y);
			Matrix mul1=(xt.times(x)).plus(lam);
			mul1=mul1.inverse();
			Matrix w=mul1.times(mul2);
			//w.print(1, 6);
			//System.out.println(w.getRowDimension());
			Matrix xtest=new Matrix(featurestest);
			Matrix ytest = new Matrix(labelstest);
			//Matrix wote=new Matrix(74, 1, w0);
			Matrix mul3 = (xtest.times(w));
			//System.out.println(mul3.getColumnDimension()+"  "+mul3.getRowDimension());
			Matrix error= mul3.minus(ytest);
			Matrix fete=(error.transpose().times(error)).times(1);
			//fete.print(1, 6);
			System.out.println("lambda  "+lambda);
			System.out.println("Testing error:"+fete.get(0, 0)/74.0);
			cte=fete.get(0, 0)/74.0;
			//error.print(1, 6);
			//Matrix wotr=new Matrix(433,1, w0);
			Matrix mul4 = (x.times(w));
			Matrix errortrain= mul4.minus(y);
			Matrix fet=(errortrain.transpose().times(errortrain)).times(1);
			//fet.print(1, 6);
			System.out.println("Training error:"+fet.get(0, 0)/433.0);
			ctr=fet.get(0, 0)/433.0;
			if(lambda==0.00001)
				gte=cte;
			if(lambda>10000)
				break;
			else
			{
				gte=cte;
				gtr=ctr;
				lambda*=10;
			}
		}
		System.out.println(lambda);
		System.out.println(gtr);
	}

}
