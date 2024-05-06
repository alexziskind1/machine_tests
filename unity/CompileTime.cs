using UnityEngine;
using UnityEditor;

//HOW TO USE
//1. Drag Script into Project (name the file CompileTime.cs)
//2. Open the window by going to Window/Analysis/CompileTime
//3. Drag/Dock the window next to the hiearchy (or somewhere else where it is visible) !important
//4. Create an new script and wait for the Unity to become responsive again, the result will be displayed in the window and console log
class CompileTime : EditorWindow
{
    bool isTrackingTime;
    double startTime, finishTime, compileTime;

    [MenuItem("Window/Analysis/Compile Times")]
    public static void Init()
    {
        CompileTime window = (CompileTime)EditorWindow.GetWindow(typeof(CompileTime));
        window.Show();
    }

    void OnGUI()
    {

        GUIStyle style = new GUIStyle(GUI.skin.label);
        style.fontStyle = FontStyle.Bold;

        // Center both lines horizontally and vertically
        GUILayout.BeginArea(new Rect(0, 0, position.width, position.height));
        GUILayout.FlexibleSpace();
        GUILayout.BeginVertical();
        // Center the label horizontally
        GUILayout.BeginHorizontal();
        GUILayout.FlexibleSpace();
        GUILayout.Label("Script Compilation Time Result: " + compileTime.ToString("0.000") + "s", style);
        GUILayout.FlexibleSpace();
        GUILayout.EndHorizontal();

        GUILayout.EndVertical();
        GUILayout.FlexibleSpace();
        GUILayout.EndArea();
    }

    void Update()
    {
        if (EditorApplication.isCompiling && !isTrackingTime)
        {
            startTime = EditorApplication.timeSinceStartup;
            isTrackingTime = true;
        }
        else if (!EditorApplication.isCompiling && isTrackingTime)
        {
            finishTime = EditorApplication.timeSinceStartup;
            isTrackingTime = false;

            compileTime = finishTime - startTime;

            Debug.Log("Script compilation time: " + compileTime.ToString("0.000") + "s");

            // Repaint the window to update the GUI with the new compile time
            Repaint();
        }
    }
}
