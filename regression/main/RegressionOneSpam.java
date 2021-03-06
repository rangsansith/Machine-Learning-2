package hw1.regression.main;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;

import Jama.Matrix;

public class RegressionOneSpam {

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
		double septhresh=0.5,tacc=0.0,tracc=0.0,temptacc=0.0,temptracc=0.0;
		int cctrind=0,cctind=0;
		ArrayList<Double[]> confusiontc= new ArrayList<Double[]>();
		ArrayList<Double[]> confusiontrc= new ArrayList<Double[]>();

		Matrix x = new Matrix(features);
		Matrix y = new Matrix(labels);
		Matrix xt = x.transpose();
		Matrix mul2=xt.times(y);
		Matrix mul1=xt.times(x);
		mul1=mul1.inverse();
		Matrix w=mul1.times(mul2);
		Matrix xtest=new Matrix(featurestest);
		Matrix ytest = new Matrix(labelstest);
		Matrix mul3 = xtest.times(w);
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
		Matrix error= mul3.minus(ytest);
		Matrix fete=(error.transpose().times(error)).times(1);
		System.out.println("Testing error:"+fete.get(0, 0)/(double) teslen);
		System.out.println("Testing Accuracy:"+tacc);
		System.out.println("CC TP:"+confusiontc.get(cctind)[0]+" FP:"+confusiontc.get(cctind)[1]+" TN:"+confusiontc.get(cctind)[2]+" FN:"+confusiontc.get(cctind)[3]);
		Matrix mul4 = x.times(w);
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
		Matrix errortrain= mul4.minus(y);
		Matrix fet=(errortrain.transpose().times(errortrain)).times(1);
		System.out.println("Training error:"+fet.get(0, 0)/(double) trlen);
		System.out.println("Training Accuracy:"+tracc);
		System.out.println("CC TP:"+confusiontrc.get(cctrind)[0]+" FP:"+confusiontrc.get(cctrind)[1]+" TN:"+confusiontrc.get(cctrind)[2]+" FN:"+confusiontrc.get(cctrind)[3]);try {
			FileWriter cclr = new FileWriter("result/spam_cclr");
			BufferedWriter cclrbw = new BufferedWriter(cclr);
			for (int i = 0; i < confusiontc.size(); i++) {
				cclrbw.write(i+" "+confusiontc.get(i)[0]+" "+confusiontc.get(i)[1]+" "+confusiontc.get(i)[2]+" "+confusiontc.get(i)[3]+"\n");
			}
			cclrbw.close();
			cclr.close();
		}catch (Exception e){
			System.out.println(e.getMessage());	
		}
	}

}
