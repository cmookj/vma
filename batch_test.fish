#!/opt/homebrew/bin/fish

for i in (seq 10)
  set output "result_$i.txt"
  MEM_LIMIT=32 cmake-test -r 1 > $output
end
