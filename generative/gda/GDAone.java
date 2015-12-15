package hw3.generative.gda;

import java.io.BufferedReader;
import java.io.FileReader;

import javax.print.DocFlavor;

import Jama.Matrix;

public class GDAone {

	private static Double[] getMean(double[][] features,int x,int y){
		Double[] mean=new Double[y];
		for (int i = 0; i < y; i++) {
			mean[i]=0.0;
			for (int j = 0; j < x; j++) {
				mean[i]+=features[j][i];
			}
			mean[i]/=x;
		}
		return mean;
	}
	private static Double[] getStd(double[][] features,int x,int y,Double[] mean){
		Double[] std=new Double[y];
		for (int i = 0; i < y; i++) {
			std[i]=0.0;
			for (int j = 0; j < x; j++) {
				std[i]+=Math.pow(features[j][i]-mean[i],2);
			}
			std[i]=Math.sqrt(std[i]/x);
		}
		return std;
	}
	private static void normalize(double[][] features,int x,int y,Double[] mean,Double[] std){
		for (int i = 0; i < y; i++) {
			for (int j = 0; j < x; j++) {
				features[j][i]-=mean[i];
				features[j][i]/=std[i];
			}
		}
	}
	private static void noramalizeboth(double[][] features,double[][] featurestest,int trx,int tex,int y){
		Double[] mean=getMean(features, trx, y);
		Double[] std=getStd(features, trx, y, mean);
		normalize(features, trx, y, mean, std);
		normalize(featurestest, tex, y, mean, std);
	}
	private static Matrix getMean(double[][] x)
	{
		int row=x.length,col=x[0].length;
		double[] mean =new double[col];
		for (int i = 0; i < col; i++) {
			mean[i]=0;
			for (int j = 0; j < row; j++) {
				mean[i]+=x[j][i];
			}
			mean[i]/=(double) row;
		}
		return new Matrix(mean,1);
	}
	private static Matrix getCovariance(double[][] x,Matrix mu){
		Matrix covariance=new Matrix(mu.getColumnDimension(), mu.getColumnDimension());
		for(int i=0;i<x.length;i++)
		{
			Matrix xtemp=new Matrix(x[i],1);
			Matrix xmu=xtemp.minus(mu);
			Matrix temp=(xmu.transpose()).times(xmu);
			covariance.plusEquals(temp);
		}
		covariance.timesEquals(1/((double) x.length));
		return covariance;
	}
	/*private static Matrix getCovariance(double[][] x1,double[][] x2,Matrix mu1,Matrix mu2){
		Matrix covariance=new Matrix(mu1.getColumnDimension(), mu1.getColumnDimension());
		for(int i=0;i<x1.length;i++)
		{
			Matrix xtemp=new Matrix(x1[i],1);
			Matrix xmu=xtemp.minus(mu);
			Matrix temp=(xmu.transpose()).times(xmu);
			covariance.plusEquals(temp);
		}
		covariance.timesEquals(1/((double) x.length));
		dispMAtrix(covariance);
		return covariance;
	}*/
	private static void dispMAtrix(Matrix a){
		for (int i = 0; i < a.getRowDimension(); i++) {
			for (int j = 0; j < a.getColumnDimension(); j++) {
				System.out.print(a.get(i, j)+"  ");
			}
			System.out.println();
		}
	}
	public static void main(String[] args) {
		int trlen=3680,teslen=921,featlen=57;
		double[][] features = new double[trlen][featlen];
		double [] labels = new double[trlen];
		double [][] featurestest= new double[teslen][featlen];
		double [] labelstest= new double[teslen];
		try {
			FileReader featureread = new FileReader("data/spam_train");
			BufferedReader featurereadbr = new BufferedReader(featureread);
			String sCurrentLine;
			String[] feats;
			int ind=0;
			while ((sCurrentLine = featurereadbr.readLine()) != null) {
				sCurrentLine = sCurrentLine.trim().replaceAll(" +", " ");
				feats=sCurrentLine.split(" ");
				labels[ind]=Double.parseDouble(feats[feats.length-1]);
				for(int i=0;i<feats.length-2;i++)
				{
					features[ind][i]=Double.parseDouble(feats[i]);
				}
				ind++;
			}
			featurereadbr.close();
			featureread.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		try {
			FileReader featureread = new FileReader("data/spam_test");
			BufferedReader featurereadbr = new BufferedReader(featureread);
			String sCurrentLine;
			String[] feats;
			int ind=0;
			while ((sCurrentLine = featurereadbr.readLine()) != null) {
				sCurrentLine = sCurrentLine.trim().replaceAll(" +", " ");
				feats=sCurrentLine.split(" ");
				for(int i=0;i<feats.length-2;i++)
				{
					featurestest[ind][i]=Double.parseDouble(feats[i]);
				}
				labelstest[ind]=Double.parseDouble(feats[feats.length-1]);
				ind++;
			}
			featurereadbr.close();
			featureread.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		noramalizeboth(features, featurestest, trlen, teslen, featlen);
		int classno=2;
		int classcount[]=new int[classno];
		int classtype[]={0,1};
		double phi=0.0;
		double[][] mu=new double[classno][featlen];

		// Calculating Phi
		for (int j = 0; j < trlen; j++) {
			if(labels[j]==1)
			{
				phi++;
			}
		}
		phi/=trlen;

		//Calculating mean for all labels
		for (int i = 0; i < classno; i++) {
			classcount[i]=0;
			for (int j = 0; j < trlen; j++) {
				if(labels[j]==classtype[i])
					classcount[i]++;
			}
		}
		for (int i = 0; i < classno; i++) {
			for (int j = 0; j < featlen; j++) {
				mu[i][j]=0.0;
				for (int k = 0; k < trlen; k++) {
					if(features[k][j]==classtype[i])
						mu[i][j]++;
				}
				mu[i][j]/=classcount[i];
			}
		}
		// Create covariance and cross covariance matrix
		Matrix covariance[] =new Matrix[classno];
		Matrix mumat[]=new Matrix[classno];
		for (int i = 0; i < classno; i++) {
			mumat[i]=new Matrix(mu[i], 1);
			covariance[i]=getCovariance(features, mumat[i]);
		}
		Matrix crosscovariance = getCovariance(features, getMean(features));
        // System.out.println(crosscovariance.det());
		double common=Math.pow(2*Math.PI,((double) featlen)/2.0);
		double denom[]=new double[classno];
		for (int i = 0; i < classno; i++) {
			denom[i]=1/(common*Math.sqrt(covariance[i].det()));
		}
		double denomcc=1/(common*Math.sqrt(crosscovariance.det()));
		System.out.println(covariance[1].det()+"   "+denom[0]+"  "+common);
		//System.out.println(covariance1.det()+"  "+detc1+"  "+detc01+" "+((double) featlen)/2.0);

		Matrix inversecovariance[]=new Matrix[classno];
		for (int i = 0; i < classno; i++) {
			inversecovariance[i]=covariance[i].inverse();
		}
		Matrix icrosscovariance = crosscovariance.inverse();

		double[] predictionsep=new double[teslen];
		double[] predictioncom=new double[teslen];

		double accsep=0.0,acccom=0.0;
		double y[]={Math.pow(phi, 1),Math.pow(1-phi, 1)};

		for (int i = 0; i < labelstest.length; i++) {
			Matrix x=new Matrix(featurestest[i], 1);
			double maxsep=-1.0,maxcom=-1.0;
			int sepi=0,comi=0;
			for (int j = 0; j < classno; j++) {
				Matrix xmu=x.minus(mumat[j]);
				double pxy=denom[j]*Math.exp(-0.5*((xmu.times(inversecovariance[j])).times((xmu.transpose())).det()));
				pxy*=y[j];
				double cpxy=denomcc*Math.exp(-0.5*((xmu.times(icrosscovariance)).times((xmu.transpose())).det()));
				cpxy*=y[j];
				if(maxsep<pxy)
				{
					maxsep=pxy;
					sepi=j;
				}
				if(maxcom<cpxy)
				{
					maxcom=cpxy;
					comi=j;
				}
			}
			predictioncom[i]=classtype[comi];
			predictionsep[i]=classtype[sepi];
			if(predictioncom[i]==labelstest[i])
				acccom++;
			if(predictionsep[i]==labelstest[i])
				accsep++;
			//System.out.println(predictioncom[i]+"  "+predictionsep[i]+"  "+labelstest[i][0]);
		}
		acccom=100.0*acccom/(double) teslen;
		accsep=100.0*accsep/(double) teslen;
		System.out.println(1-phi);
		System.out.println("Accuracy Common Covariance: "+acccom);
		System.out.println("Accuracy separate Covariance: "+accsep);
		acccom=0.0;accsep=0.0;
		predictionsep=new double[trlen];
		predictioncom=new double[trlen];
		for (int i = 0; i < labels.length; i++) {
			Matrix x=new Matrix(features[i], 1);
			double maxsep=-1.0,maxcom=-1.0;
			int sepi=0,comi=0;
			for (int j = 0; j < classno; j++) {
				Matrix xmu=x.minus(mumat[j]);
				double pxy=denom[j]*Math.exp(-0.5*((xmu.times(inversecovariance[j])).times((xmu.transpose())).det()));
				pxy*=y[j];
				double cpxy=denomcc*Math.exp(-0.5*((xmu.times(icrosscovariance)).times((xmu.transpose())).det()));
				cpxy*=y[j];
				if(maxsep<pxy)
				{
					maxsep=pxy;
					sepi=j;
				}
				if(maxcom<cpxy)
				{
					maxcom=cpxy;
					comi=j;
				}
			}
			predictioncom[i]=classtype[comi];
			predictionsep[i]=classtype[sepi];
			if(predictioncom[i]==labels[i])
				acccom++;
			if(predictionsep[i]==labels[i])
				accsep++;
			//System.out.println(predictioncom[i]+"  "+predictionsep[i]+"  "+labelstest[i][0]);
		}
		acccom=100.0*acccom/(double) trlen;
		accsep=100.0*accsep/(double) trlen;
		System.out.println("Accuracy Common Covariance: "+acccom);
		System.out.println("Accuracy separate Covariance: "+accsep);
	}

}
