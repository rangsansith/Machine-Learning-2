package hw6.svm.smo;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;

public class TwoClassAllNew {

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
	static HashMap<Integer, Boolean> nonbound = new HashMap<Integer, Boolean>();
	static ArrayList<Integer> nonboundlist = new ArrayList<Integer>();
	static HashMap<Integer, Double> errorcache = new HashMap<Integer, Double>();
	static TreeMap<Double, Integer> sortederrorcacheplus = new TreeMap<Double, Integer>();
	static TreeMap<Double, Integer> sortederrorcacheminus = new TreeMap<Double, Integer>();
	static double[] alpha;
	static double b;
	static double regc=1;
	static double numtol=0.001;
	static int maxpass=10;
	static int nonboundcount;
	static Random jselect = new Random(); 
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
	private static void initalphas(){
		for (int i = 0; i < trainlen; i++) {
			alpha[i]=0.0;
		}
	}
	private static void kernel(){
		for(int i=0;i<trainlen;i++)
		{
			for(int j=0;j<trainlen;j++)
			{
				double ans=0;
				for(int k=0;k<featlen;k++)
				{
					ans+=featurestrain[i][k]*featurestrain[j][k];
				}
				kernelmat[i][j]=ans;
			}
		}
	}
	private static double mpredict(int xno){
		double ans=b;
		for (int i = 0; i < trainlen; i++) {
			ans+=alpha[i]*kernelmat[i][xno]*labelstrain[i];
		}
		return ans;
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
	private static void updatenonbound(int i,double val){
		if(val<regc&&val>0)
		{
			if(!nonbound.containsKey(i))
			{
				nonbound.put(i, true);
				nonboundlist.add(i);
			}
		}
		else
		{
			if(nonbound.containsKey(i))
			{
				nonbound.remove(i);
				nonboundlist.remove((Integer)i);
			}
		}
	}
	private static void storecache(int k,double error){
		if(errorcache.containsKey(k))
		{
			double d=errorcache.get(k);
			errorcache.put(k, error);
			if(d>0)
			{
				sortederrorcacheplus.remove(-d);
				sortederrorcacheplus.put(-error, k);
			}
			else
			{
				sortederrorcacheminus.remove(d);
				sortederrorcacheminus.put(error, k);
			}
		}
		else
		{
			errorcache.put(k, error);
			if(error>0)
			{
				sortederrorcacheplus.put(-error, k);
			}
			else
			{	
				sortederrorcacheminus.put(error, k);
			}
		}

	}
	private static void setcache(){
		for (int i = 0; i < trainlen; i++) {
			storecache(i, mpredict(i)-labelstrain[i]);
		}
	}
	private static int update(int i,int j,double ei){
		if(i==j)
			return 0;
		double ej=mpredict(j)-labelstrain[j];
		storecache(j, ej);
		double alphai = alpha[i];
		double alphaj = alpha[j];
		double[] LH = LH(i, j);
		if(LH[0]==LH[1])
			return 0;
		double eta=2*kernelmat[i][j]-kernelmat[i][i]-kernelmat[j][j];
		if(eta>=0)
			return 0;
		alpha[j]=alphaj-labelstrain[j]*(ei-ej)/eta;
		if(alpha[j]>LH[1])
			alpha[j]=LH[1];
		else if(alpha[j]<LH[0])
			alpha[j]=LH[0];
		if(Math.abs(alphaj-alpha[j])<0.001)
			return 0;
		alpha[i]=alpha[i]+labelstrain[i]*labelstrain[j]*(alphaj-alpha[j]);
		updatenonbound(j,alpha[j]);
		updatenonbound(i,alpha[i]);
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
	private static int quickfindmax(double ei){
		if(errorcache.size()<trainlen)
			setcache();
		if(ei>0)
		{
			for (Double key:sortederrorcacheplus.keySet()) {
				int d=sortederrorcacheplus.get(key);
				if(nonbound.containsKey(d))
					return d;
			}	
		}
		else
		{
			for (Double key:sortederrorcacheminus.keySet()) {
				int d=sortederrorcacheminus.get(key);
				if(nonbound.containsKey(d))
					return d;
			}
		}
		return 0;
	}
	private static int selectfirstalpha(int i){
		nonboundcount=nonbound.size();
		double alphai=alpha[i];
		double yi=labelstrain[i];
		double ei=mpredict(i)-yi;
		storecache(i, ei);
		double ri=ei*yi;
		int j=-1;
		if(((ri<-numtol)&&(alphai<regc))||((ri>numtol)&&(alphai>0)))
		{
			if(nonboundcount>1)
			{
				j=quickfindmax(ei);
				if (update(i, j,ei)==1)
				return 1;
				j=jselect.nextInt(nonboundcount);
				j=nonboundlist.get(j);
				if (update(i, j,ei)==1)
					return 1;
			}
			j=jselect.nextInt(trainlen);
			if (update(i, j,ei)==1)
			return 1;
		}
		return 0;
	}
	public static void main(String[] args) {

		readData();
		int folds=10;
		uniformSplits(folds);
		pickKfold(1);
		writeforlib();
		labelstrain=changeleabels(labelstrain);
		alpha=new double[trainlen];
		kernelmat=new double[trainlen][trainlen];
		kernel();
		initalphas();
		b=0;
		nonboundcount=0;
		int passes=0;
		int tot=0;
		int numchanged=0;
		boolean allcheck=true;
		while(numchanged>0 || allcheck){
			tot++;
			if(tot>1000)
				break;
			System.out.println(tot+"  "+nonboundcount+"  "+numchanged);
			numchanged=0;
			//setcache();
			if(allcheck){
				for (int i = 0; i < trainlen; i++) {
					numchanged+=selectfirstalpha(i);
				}
			}
			else
			{
				HashMap<Integer, Boolean> temp = new HashMap<Integer,Boolean>(nonbound);
				for (Integer key : temp.keySet()) {
					numchanged+=selectfirstalpha(key);
					temp = new HashMap<Integer,Boolean>(nonbound);
				}
			}
			if(allcheck)
				allcheck=false;
			else if(numchanged==0)
				{
				 allcheck=true;
				 passes++;
				 double acc=0.0;
				 for (int i = 0; i < trainlen; i++) {
						//System.out.println(mpredict(i)+" "+labelstrain[i]);
						if(mpredict(i)*labelstrain[i]>0)
							acc++;

					}
					System.out.println("Accuracy: "+acc/(double) trainlen);
				}
			if(passes>maxpass)
				break;
		}
		double acc=0.0;
		for (int i = 0; i < trainlen; i++) {
			System.out.println(mpredict(i)+" "+labelstrain[i]);
			if(mpredict(i)*labelstrain[i]>0)
				acc++;

		}
		System.out.println(acc/(double) trainlen);
	}

}
