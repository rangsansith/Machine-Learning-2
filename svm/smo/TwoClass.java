package hw6.svm.smo;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

public class TwoClass {

	static int len=4601;
	static int featlen=57;
	static int classno=2;
	static int trainlen,testlen;
	static double[][] features = new double[len][featlen];
	static double[] labels = new double[len];
	static ArrayList<double[][]> featklist = new ArrayList<double[][]>();
	static ArrayList<double[]> labelsklist = new ArrayList<double[]>();
	static ArrayList<Integer> indexlist = new ArrayList<Integer>();
	static double[][] featurestrain = new double[1][featlen];
	static double [] labelstrain = new double[1];
	static double [][] featurestest= new double[1][featlen];
	static double [][] kernelmat= new double[1][1];
	static double [] labelstest= new double[1];
	static double[] alpha;
	static double b;
	static double regc=10;
	static double numtol=0.001;
	static int maxpass=10;
	static int kerneltype=2;
	static double sigmag=1.0/featlen;	
	static double tanc=0.1;
	static double tanb=0.0;
	private static void readData()
	{
		int one=0,zero=0;
		try {
			FileReader featureread = new FileReader("data/spambase.data");
			BufferedReader featurereadbr = new BufferedReader(featureread);
			String sCurrentLine;
			String[] feats;
			int ind=0;
			while ((sCurrentLine = featurereadbr.readLine()) != null) {
				indexlist.add(ind);
				feats=sCurrentLine.split(",");
				for(int i=0;i<feats.length-1;i++)
				{
					features[ind][i]=Double.parseDouble(feats[i]);

				}
				labels[ind]=Double.parseDouble(feats[feats.length-1]);
				if(labels[ind]==0.0)
					zero++;
				else
					one++;
				ind++;
			}
			featurereadbr.close();
			featureread.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		//System.out.println(zero+"   "+one+"  "+(double) zero/(zero+one));
	}
	private static void randSplits(int folds)
	{
		HashMap<Integer, Boolean> seen = new HashMap<Integer, Boolean>();
		int kfold=len/folds,listno=0,one=0,zero=0;
		Random n = new Random(); 
		while(indexlist.size()!=0){
			one=0;zero=0;
			int r=Math.min(kfold, indexlist.size());
			double[][] featurestemp = new double[r][featlen];
			double [] labelstemp = new double[r];
			for (int i = 0; i < kfold; i++) {
				if(indexlist.size()==0)
					break;
				int ind=n.nextInt(indexlist.size());
				int tmpind=indexlist.get(ind);
				if(seen.containsKey(tmpind))
					System.out.println("Error: bad Coder");
				else
					seen.put(tmpind, true);
				for (int j = 0; j < featlen; j++) {
					featurestemp[i][j]=features[tmpind][j];
				}
				labelstemp[i]=labels[tmpind];
				if(labels[tmpind]==0.0)
					zero++;
				else if(labels[tmpind]==1.0)
					one++;
				else
					System.out.println("Error: bad Coder");

				indexlist.remove(ind);
			}
			//System.out.println(zero+"   "+one+"   "+(double) zero/(zero+one));
			featklist.add(listno,featurestemp);
			labelsklist.add(listno,labelstemp);
			listno++;
		}
	}
	private static void uniformSplits(int folds){
		ArrayList<ArrayList<double[]>> tft= new ArrayList<ArrayList<double[]>>();
		ArrayList<ArrayList<Double>> tlt= new ArrayList<ArrayList<Double>>();
		for (int i = 0; i < folds; i++) {
			tft.add(new ArrayList<double[]>());
			tlt.add(new ArrayList<Double>());
		}
		for (int i = 0; i < len; i++) {
			ArrayList<double[]> temp = tft.get(i%folds);
			temp.add(features[i]);
			tft.set(i%folds, temp);
			ArrayList<Double> lemp = tlt.get(i%folds);
			lemp.add(labels[i]);
			tlt.set(i%folds,lemp);
		}
		if(tft.size()!=tlt.size())
			System.out.println("Error: BadCode");
		int one=0,zero=0,listno=0;
		for (int i = 0; i < tft.size(); i++) {
			one=0;zero=0;
			double[][] featurestemp = new double[tft.get(i).size()][featlen];
			double [] labelstemp = new double[tlt.get(i).size()];
			for (int j = 0; j < tft.get(i).size(); j++) {
				featurestemp[j]=tft.get(i).get(j);
				labelstemp[j]=tlt.get(i).get(j);
				if(labelstemp[j]==0)
					zero++;
				else
					one++;
			}
			//System.out.println(zero+"   "+one+"   "+(double) zero/(zero+one));
			featklist.add(listno,featurestemp);
			labelsklist.add(listno,labelstemp);
			listno++;
		}
	}
	private static void pickKfold(int k){
		k--;
		testlen=labelsklist.get(k).length;
		trainlen=len-testlen;
		featurestrain=new double[trainlen][featlen];
		labelstrain=new double[trainlen];
		featurestest=new double[testlen][featlen];
		labelstest=new double[testlen];
		int ind=0;
		for (int i = 0; i < featklist.size(); i++) {
			if(i==k)
			{
				featurestest=featklist.get(k);
				labelstest=labelsklist.get(k);
				continue;
			}
			double[][] tf = featklist.get(i);
			double[] tl = labelsklist.get(i);
			for (int j = 0; j < tl.length; j++) {
				featurestrain[ind]=tf[j];
				labelstrain[ind]=tl[j];
				ind++;
			}
		}
	}
	private static double[] changeleabels(double[] labelsc){
		for (int i = 0; i < labelsc.length; i++) {
			if(labelsc[i]==0)
				labelsc[i]=-1;
		}
		return labelsc;
	}
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
	private static void setkernel(){
		for(int i=0;i<trainlen;i++)
		{
			for(int j=0;j<trainlen;j++)
			{
				kernelmat[i][j]=kernel(featurestrain[i],featurestrain[j]);
			}
		}
	}
	private static double mpredict(int xno){
		double ans=b;
		double []kern=kernelmat[xno];
		for (int i = 0; i < trainlen; i++) {
			ans+=alpha[i]*kern[i]*labelstrain[i];
		}
		return ans;
	}
	private static double mpredict(double [] xno){
		double ans=b;
		for (int i = 0; i < trainlen; i++) {
			ans+=alpha[i]*kernel(xno,featurestrain[i])*labelstrain[i];
		}
		return ans;
	}
	private static void initalphas(){
		for (int i = 0; i < trainlen; i++) {
			alpha[i]=0.0;
		}
	}
	private static double[] LH(int i,int j){
		double[] ans={0,0};
		if(labelstrain[i]==labelstrain[j])
		{
			ans[0]=Math.max(0, alpha[j]+alpha[i]-regc);
			ans[1]=Math.min(regc,alpha[j]+alpha[i]);
		}
		else
		{
			ans[0]=Math.max(0, alpha[j]-alpha[i]);
			ans[1]=Math.min(regc, regc+alpha[j]-alpha[i]);
		}
		return ans;
	}
	private static void writeforlib(){

		try {
			FileWriter fw = new FileWriter("resulthw6/trainlib.txt");
			BufferedWriter bw = new BufferedWriter(fw);
			for (int i = 0; i < trainlen; i++) {
				bw.write(((int)labelstrain[i])+"");
				for (int j = 0; j < featlen; j++) {
					bw.write(" "+(j+1)+":"+featurestrain[i][j]);
				}
				bw.newLine();
			}
			bw.close();
			fw.close();

		}catch (Exception e){
			System.out.println(e.getMessage());	
		}
		try {
			FileWriter fw = new FileWriter("resulthw6/testlib.txt");
			BufferedWriter bw = new BufferedWriter(fw);
			for (int i = 0; i < testlen; i++) {
				bw.write(((int)labelstest[i])+"");
				for (int j = 0; j < featlen; j++) {
					bw.write(" "+(j+1)+":"+featurestest[i][j]);
				}
				bw.newLine();
			}
			bw.close();
			fw.close();

		}catch (Exception e){
			System.out.println(e.getMessage());	
		}
	}
	private static int update(int i,int j,double ei){
		double ej=mpredict(j)-labelstrain[j];
		double alphai = alpha[i];
		double alphaj = alpha[j];
		double[] LH = LH(i, j);
		if(LH[0]==LH[1])
			return 0;
		double eta=2*kernelmat[i][j]-kernelmat[i][i]-kernelmat[j][j];
		if(eta>=0)
			return 0;
		alpha[j]=alpha[j]-labelstrain[j]*(ei-ej)/eta;
		if(alpha[j]>LH[1])
			alpha[j]=LH[1];
		else if(alpha[j]<LH[0])
			alpha[j]=LH[0];
		if(Math.abs(alphaj-alpha[j])<0.00001)
			return 0;
		alpha[i]=alpha[i]+labelstrain[i]*labelstrain[j]*(alphaj-alpha[j]);
		double b1=b-ei-labelstrain[i]*(alpha[i]-alphai)*kernelmat[i][i]-labelstrain[j]*(alpha[j]-alphaj)*kernelmat[i][j];
		double b2=b-ej-labelstrain[i]*(alpha[i]-alphai)*kernelmat[i][j]-labelstrain[j]*(alpha[j]-alphaj)*kernelmat[j][j];
		if(alpha[i]<regc)
			b=b1;
		else if(alpha[j]<regc)
			b=b2;
		else
			b=(b1+b2)/2;
		return 1;
	}
	private static void SMOng(){
		alpha=new double[trainlen];
		kernelmat=new double[trainlen][trainlen];
		setkernel();
		initalphas();
		b=0;
		int passes=0;
		int tot=0;
		while(passes<maxpass){
			tot++;
			//System.out.println(tot);
			int numalpha=0;

			for (int i = 0; i < trainlen; i++) {
				double ei=mpredict(i)-labelstrain[i];
				if(((labelstrain[i]*ei<-numtol)&&(alpha[i]<regc))||((labelstrain[i]*ei>numtol)&&(alpha[i]>0))){
					int j = (int)Math.floor(Math.random()*(trainlen-1));
					j = (j<i)?j:(j+1);

					numalpha+=update(i, j,ei);
				}
			}
			//System.out.println(tot+"  "+numalpha);
			if(numalpha==0)
				passes++;
			else
				passes=0;
			if(tot==1000)
				break;
		}
	}
	public static void main(String[] args) {

		readData();
		int folds=10;
		uniformSplits(folds);
		for (int k = 1; k <= 10; k++) {
			
		pickKfold(k);
		labelstrain=changeleabels(labelstrain);
		labelstest=changeleabels(labelstest);
		//writeforlib();
		
		SMOng();
		double acc=0.0;
		for (int i = 0; i < trainlen; i++) {
			//System.out.println(mpredict(i)+" "+labelstrain[i]+" "+alpha[i]);
			if(mpredict(i)*labelstrain[i]>0)
				acc++;

		}
		System.out.println("train:"+acc/(double) trainlen);
		System.out.println("train:"+acc/(double) trainlen);
		acc=0.0;
		for (int i = 0; i < trainlen; i++) {
			//System.out.println(mpredict(i)+" "+labelstrain[i]+" "+alpha[i]);
			if(mpredict(featurestrain[i])*labelstrain[i]>0)
				acc++;

		}
		System.out.println("trainaga:"+acc/(double) trainlen);
		labelstest=changeleabels(labelstest);
		acc=0.0;
		for (int i = 0; i < testlen; i++) {
			//System.out.println(mpredict(i)+" "+labelstrain[i]+" "+alpha[i]);
			if(mpredict(featurestest[i])*labelstest[i]>0)
				acc++;

		}
		System.out.println("test:"+acc/(double) testlen);
		}
	}

}
