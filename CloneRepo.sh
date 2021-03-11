function createBareRepo {
	TargetRepoLocation="https://github.com/mlguy001/gitTest"
	
	WorkspaceLocation="C:\Users\Nidhal\Desktop\Classes\Git testing"
	#WorkspaceName="Workspace $FolderName"
	WorkspaceName="WorkspaceGitTest"
	
			
			
				git clone "$TargetRepoLocation/" "$WorkspaceLocation/$WorkspaceName/"


}

createBareRepo

read -p ""