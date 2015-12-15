package hw2.regression.logistic;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;

import Jama.Matrix;

public class LogisticOneSpam {

	private static double hofw(double w[],double x[],double y){
		double ans=0.0;
		for (int i = 0; i < w.length; i++) {
			ans+=w[i]*x[i];
			//System.out.println(ans+"    "+w[i]+"    "+x[i]);
		}
		//ans+=w0;
		//System.out.println(ans);
		ans=y-(1.0/(1.0+Math.exp(-ans)));
		return ans;
	}
	public static void main(String[] args) {
		int trlen=3680,teslen=921,featlen=58;
		double[][] features = new double[trlen][featlen];
		double [][] labels = new double[trlen][1];
		double[] mean= new double[featlen];
		double[] std= new double[featlen];
		for(int i=0;i<featlen;i++)
		{
			mean[i]=0.0;
		}
		FileReader featureread;
		try {
			featureread = new FileReader("data/spam_train");
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
				labels[ind][0]=Double.parseDouble(feats[feats.length-1]);
				ind++;
			}
			featurereadbr.close();
			featureread.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		for(int i=0;i<featlen;i++)
		{
			mean[i]/=(double) trlen;
		}
		for (int i = 0; i < mean.length; i++) {
			std[i]=0.0;
			for (int j = 0; j < features.length; j++) {
				std[i]+=(features[j][i]-mean[i])*(features[j][i]-mean[i]);
			}
			std[i]=Math.sqrt(std[i]);
		}
		for (int i = 0; i < features.length; i++) {
			for (int j = 0; j < mean.length-1; j++) {
				features[i][j]-=mean[j];
				features[i][j]/=std[j];
			}
		}
		double [][] featurestest= new double[teslen][featlen];
		double [][] labelstest= new double[teslen][1];
		try {
			featureread = new FileReader("data/spam_test");
			BufferedReader featurereadbr = new BufferedReader(featureread);
			String sCurrentLine;
			String[] feats;
			int ind=0;
			while ((sCurrentLine = featurereadbr.readLine()) != null) {
				sCurrentLine = sCurrentLine.trim().replaceAll(" +", " ");
				feats=sCurrentLine.split(" ");
				for(int i=0;i<feats.length-2;i++)
				{
					featurestest[ind][i]=(Double.parseDouble(feats[i])-mean[i])/std[i];
				}
				featurestest[ind][feats.length-2]=Double.parseDouble(feats[feats.length-2]);
				labelstest[ind][0]=Double.parseDouble(feats[feats.length-1]);
				ind++;
			}
			featurereadbr.close();
			featureread.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		double lambda=10.01;
		double w[]=new double[featlen];
		for (int ij = 0; ij < w.length; ij++) {
			w[ij]=100;
		}
		double gte=100000.0,cte=0.0,ctr=0.0,thresh=0.0000001,tacc=0.0,tracc=0.0,septhresh=0.5,temptracc=0.0,temptacc=0.0;
		Matrix x = new Matrix(features);
		Matrix y = new Matrix(labels);
		Matrix xtest=new Matrix(featurestest);
		Matrix ytest = new Matrix(labelstest);
		while(true)
		{
			for (int ij = 0; ij < features.length; ij++) {

				for (int j = 0; j < w.length; j++) {
					w[j]=w[j]+lambda*hofw(w,features[ij],labels[ij][0])*features[ij][j];
				}
			}
			Matrix wma=new Matrix(w, w.length);
			Matrix mul3 = xtest.times(wma);

			for(int ij=0;ij<mul3.getRowDimension();ij++)
			{
				mul3.set(ij, 0, 1/(1+Math.exp(-mul3.get(ij, 0))));
			}

			Matrix error= ytest.minus(mul3);
			tacc=0.0;
			for (int j = 0; j < mul3.getRowDimension(); j++) {
				if( Math.floor(mul3.get(j, 0)+septhresh) == ytest.get(j, 0) )
				{
					tacc++;
				}
			}
			tacc/=(double) mul3.getRowDimension();
			tacc*=100;
			Matrix fete=(error.transpose().times(error)).times(1);
			System.out.println("Testing error:"+fete.get(0, 0)/((double) (teslen)));
			System.out.println("testing Accuracy:"+tacc);
			cte=fete.get(0, 0)/((double) (teslen));
			Matrix mul4 = x.times(wma);

			for(int ij=0;ij<mul4.getRowDimension();ij++)
			{
				mul4.set(ij, 0, 1/(1+Math.exp(-mul4.get(ij, 0))));
			}
			Matrix errortrain= y.minus(mul4);
			tracc=0.0;
			for (int j = 0; j < mul4.getRowDimension(); j++) {
				if( Math.floor(mul4.get(j, 0)+septhresh) == y.get(j, 0) )
				{
					tracc++;
				}
			}
			tracc/=(double) mul4.getRowDimension();
			tracc*=100;
			Matrix fet=(errortrain.transpose().times(errortrain)).times(1);
			System.out.println("Training error:"+fet.get(0, 0)/((double) (trlen)));
			System.out.println("training Accuracy:"+tracc);
			ctr=fet.get(0, 0)/((double) (trlen));
			if(Math.abs(cte-gte)<thresh)
			{
				//System.out.println(Math.abs(cte-gte)+"  ");
				break;
			}
			//if(cte>gte)
			//break;
			else
			{
				gte=cte;
			}

		}
		System.out.println("over  \n");
		int cctrind=0,cctind=0;
		ArrayList<Double[]> confusiontc= new ArrayList<Double[]>();
		ArrayList<Double[]> confusiontrc= new ArrayList<Double[]>();
		Matrix wma=new Matrix(w, w.length);
		Matrix mul3 = (xtest.times(wma));
		for(int ij=0;ij<mul3.getRowDimension();ij++)
		{
			System.out.println("before "+Math.floor(mul3.get(ij, 0)));
			mul3.set(ij, 0, 1/(1+Math.exp(-mul3.get(ij, 0))));
			System.out.println("after "+Math.floor(mul3.get(ij, 0)));
		}
		Matrix error= ytest.minus(mul3);
		Matrix fete=(error.transpose().times(error)).times(1);
		septhresh=0.0;
		for(int k=0; k < mul3.getRowDimension() ; k++)
		{
			septhresh=Math.min(septhresh, mul3.get(k, 0));
		}
		septhresh-=1;
		for(int k=0; k < mul3.getRowDimension()+1 ; k++)
		{
			if(k>0)
				septhresh=mul3.get(k-1, 0);
			temptacc=0.0;
			Double[] tcc={0.0,0.0,0.0,0.0};
			for (int j = 0; j < mul3.getRowDimension(); j++) {
				if(mul3.get(j, 0)>septhresh)
				{
					if(ytest.get(j, 0)==1)
					{
						tcc[0]++;
						temptacc++;
					}
					else
					{
						tcc[1]++;	
					}
				}
				else
				{
					if(ytest.get(j, 0)==0)
					{
						tcc[2]++;
						temptacc++;
					}
					else
					{
						tcc[3]++;	
					}
				}
			}
			temptacc/=(double) mul3.getRowDimension();
			temptacc*=100;
			confusiontc.add(k,tcc);
			if(tacc<temptacc)
			{
				tacc=temptacc;
				cctind=k;
			}
		}
		System.out.println("Testing error:"+fete.get(0, 0)/(double) teslen);
		System.out.println("testing Accuracy:"+tacc);
		System.out.println("CC TP:"+confusiontc.get(cctind)[0]+" FP:"+confusiontc.get(cctind)[1]+" TN:"+confusiontc.get(cctind)[2]+" FN:"+confusiontc.get(cctind)[3]);
		Matrix mul4 = (x.times(wma));
		for(int ij=0;ij<mul4.getRowDimension();ij++)
		{
			mul4.set(ij, 0, 1/(1+Math.exp(-mul4.get(ij, 0))));
		}
		Matrix errortrain= y.minus(mul4);
		Matrix fet=(errortrain.transpose().times(errortrain)).times(1);
		septhresh=0.0;
		for(int k=0; k < mul4.getRowDimension() ; k++)
		{
			septhresh=Math.min(septhresh, mul4.get(k, 0));
		}
		septhresh-=1;
		for(int k=0; k < mul4.getRowDimension()+1 ; k++)
		{
			if(k>0)
				septhresh=mul4.get(k-1, 0);
			temptracc=0.0;
			Double[] trcc={0.0,0.0,0.0,0.0};
			for (int j = 0; j < mul4.getRowDimension(); j++) {
				if( Math.floor(mul4.get(j, 0)+septhresh) == 1)
				{
					if(y.get(j, 0)==1)
					{
						trcc[0]++;
						temptracc++;
					}
					else
					{
						trcc[1]++;	
					}
				}
				else
				{
					if(y.get(j, 0)==0)
					{
						trcc[2]++;
						temptracc++;
					}
					else
					{
						trcc[3]++;	
					}
				}
			}
			temptracc/=(double) mul4.getRowDimension();
			temptracc*=100;
			confusiontrc.add(k,trcc);
			if(tracc<temptracc)
			{
				tracc=temptracc;
				cctrind=k;
			}
		}
		System.out.println("Training error:"+fet.get(0, 0)/(double) trlen);
		System.out.println("training Accuracy:"+tracc);
		System.out.println("CC TP:"+confusiontrc.get(cctrind)[0]+" FP:"+confusiontrc.get(cctrind)[1]+" TN:"+confusiontrc.get(cctrind)[2]+" FN:"+confusiontrc.get(cctrind)[3]);
		try {
			FileWriter cclogr = new FileWriter("result/spam_cclogr");
			BufferedWriter cclogrbw = new BufferedWriter(cclogr);
			for (int i = 0; i < confusiontc.size(); i++) {
				cclogrbw.write(i+" "+confusiontc.get(i)[0]+" "+confusiontc.get(i)[1]+" "+confusiontc.get(i)[2]+" "+confusiontc.get(i)[3]+"\n");
			}
			cclogrbw.close();
			cclogr.close();
		}catch (Exception e){
			System.out.println(e.getMessage());	
		}

	}

}
